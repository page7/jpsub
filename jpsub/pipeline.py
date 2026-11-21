"""主流程编排：视频 → cue → 日文 → 中文 → ASS。"""

from __future__ import annotations

import difflib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cues import SegmentConfig, segment
from .ocr import make_backend
from .srtio import Cue, read_srt, write_ass
from .translate import make_translator
from .video import grab_band_color, imwrite, probe, stream_band_color, stream_band_gray


def _say(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class Options:
    video: Path | None = None
    srt_in: Path | None = None
    out_dir: Path | None = None
    band: tuple[float, float] = (0.70, 0.99)
    sample_fps: float = 6.0
    seg: SegmentConfig = field(default_factory=SegmentConfig)
    ocr: str = "paddle"
    ocr_model: str = "Index-Translate-35B-A3B"
    ocr_base_url: str = "https://index-translate.bilibili.com/v1"
    ocr_key: str = ""
    translator: str = "bilibili"
    tr_model: str | None = None
    tr_base_url: str | None = None
    tr_key: str | None = None
    batch_size: int = 16
    concurrency: int = 2
    retries: int = 3
    max_seconds: float | None = None
    max_line: int = 20
    debug_dir: Path | None = None


def _pbar(done: int, total: int, width: int = 28) -> str:
    frac = done / max(1, total)
    filled = int(width * frac)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total} ({frac * 100:5.1f}%)"


# ---------------------------------------------------------------- 合并/清洗


def _norm(s: str) -> str:
    return re.sub(r"[\s\u3000]+", "", s)


def merge_repeats(cues: list[Cue], sim: float = 0.88, max_gap: float = 0.40) -> list[Cue]:
    """相邻两条 OCR 文本几乎一样 → 并成一条。图像分割偶尔会把一句切两半。"""
    out: list[Cue] = []
    for c in cues:
        if not out:
            out.append(c)
            continue
        prev = out[-1]
        a, b = _norm(prev.source), _norm(c.source)
        if not a or not b:
            out.append(c)
            continue
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        short, long_ = sorted((len(a), len(b)))
        contained = short > 0 and (a in b or b in a) and short / long_ > 0.75
        if (ratio > sim or contained) and c.start - prev.end <= max_gap:
            prev.end = max(prev.end, c.end)
            continue
        out.append(c)
    return out


_BREAKS = "、。，．！？!?…‥」』）)】"


def wrap_text(text: str, limit: int) -> str:
    """超长译文按标点断成两行（字幕一条最多两行）。"""
    if limit <= 0 or len(text) <= limit:
        return text
    if len(text) > limit * 2:
        return text  # 太长不硬断，交给人工
    mid = len(text) // 2
    best = -1
    bestscore = 1e9
    for i, ch in enumerate(text):
        if ch in _BREAKS:
            s = abs(i - mid)
            if s < bestscore:
                bestscore, best = s, i + 1
    if best < 0:
        best = mid
    return text[:best].strip() + "\n" + text[best:].strip()


# ---------------------------------------------------------------- 各阶段


def _prune_cue_shots(cues: list[Cue], debug_dir: Path | None) -> int:
    """只保留最终输出 cue 的截图，删掉被合并/丢弃的 cue 截图。"""
    if not debug_dir:
        return 0
    keep: set[Path] = set()
    for c in cues:
        shot = c.extra.get("shot")
        if shot:
            try:
                keep.add(Path(shot).resolve())
            except OSError:
                pass
    for p in debug_dir.glob("cue_*.png"):
        try:
            if p.resolve() not in keep:
                p.unlink()
        except OSError:
            pass
    return len(keep)


def stage_ocr(opts: Options, info, cues: list[Cue]) -> list[Cue]:
    backend = make_backend(
        opts.ocr,
        log=_say,
        model=opts.ocr_model,
        base_url=opts.ocr_base_url,
        api_key=opts.ocr_key,
    )
    if opts.debug_dir:
        opts.debug_dir.mkdir(parents=True, exist_ok=True)
        # 同名旧截图先清掉，避免不同参数的结果混在一起
        for old in opts.debug_dir.glob("cue_*.png"):
            try:
                old.unlink()
            except OSError:
                pass
        _say(f"  cue 截图目录: {opts.debug_dir}")

    total = len(cues)
    _say(f"  逐条 OCR（{total} 条）…")
    kept: list[Cue] = []
    shot_saved = 0
    t0 = time.time()
    for i, c in enumerate(cues, 1):
        # 取 cue 中段，避开淡入淡出
        t = c.start + min(0.35, c.duration * 0.5)
        img = grab_band_color(info, t, opts.band)
        text = ""
        if img is not None:
            if opts.debug_dir:
                shot = opts.debug_dir / f"cue_{i:04d}_{t:08.2f}.png"
                if imwrite(shot, img):
                    c.extra["shot"] = str(shot)
                    shot_saved += 1
            try:
                text = backend.recognize(img)
            except Exception as e:
                _say(f"\n  ! 第 {i} 条 OCR 失败 ({t:.2f}s): {e}")
            from .ocr import detect_text_color

            c.color = detect_text_color(img, getattr(backend, "last_polys", None))
        c.source = text
        if text:
            kept.append(c)

        if i % 5 == 0 or i == total:
            el = time.time() - t0
            eta = el / i * (total - i)
            sys.stdout.write(f"\r  {_pbar(i, total)}  剩余约 {eta:4.0f}s ")
            sys.stdout.flush()
    sys.stdout.write("\n")
    if opts.debug_dir:
        _say(f"  cue 截图：已保存 {shot_saved}/{total} 张")
        if shot_saved == 0:
            _say("  ! cue 截图一张都没写成，检查输出目录是否存在/可写")
    _say(f"  OCR 完成，有效 {len(kept)}/{total} 条，耗时 {time.time() - t0:.0f}s")
    return kept


def stage_translate(opts: Options, cues: list[Cue]) -> None:
    if opts.translator == "none":
        _say("  跳过翻译（--translate none）")
        return
    tr = make_translator(
        opts.translator,
        model=opts.tr_model,
        api_key=opts.tr_key,
        base_url=opts.tr_base_url,
        batch_size=opts.batch_size,
        concurrency=opts.concurrency,
        retries=opts.retries,
        log=_say,
    )
    texts = [c.source.replace("\n", " ").strip() for c in cues]
    _say(f"  翻译 {len(texts)} 条（{opts.translator} / {tr.cfg.model}）…")
    t0 = time.time()
    last = [0]

    def prog(done: int, total: int) -> None:
        if done - last[0] >= 3 or done == total:
            last[0] = done
            sys.stdout.write(f"\r  {_pbar(done, total)} ")
            sys.stdout.flush()

    outs = tr.run(texts, progress=prog)
    sys.stdout.write("\n")
    empty = 0
    for c, o in zip(cues, outs):
        c.text = (o or "").strip()
        if not c.text:
            empty += 1
    _say(f"  翻译完成，耗时 {time.time() - t0:.0f}s，空译文 {empty} 条")


# ---------------------------------------------------------------- 入口


def _rgb_of(hexv: str) -> tuple[int, int, int]:
    h = hexv.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hue_sat(rgb: tuple[int, int, int]) -> tuple[float, float]:
    """返回 (色相角度, 饱和度 0~1)。灰阶色相无意义，返回 (-1, 0)。"""
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mx <= 0 or mx == mn:
        return -1.0, 0.0
    d = mx - mn
    if mx == r:
        h = 60 * ((g - b) / d % 6)
    elif mx == g:
        h = 60 * ((b - r) / d + 2)
    else:
        h = 60 * ((r - g) / d + 4)
    return h % 360, d / mx


def _hue_gap(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    ha, sa = _hue_sat(a)
    hb, sb = _hue_sat(b)
    if sa < 0.15 or sb < 0.15:  # 有一边是灰阶，色相比较没意义
        return 999.0
    d = abs(ha - hb)
    return min(d, 360 - d)


def _assign_index(rgb: tuple[int, int, int], centers: list[tuple[int, int, int]]) -> int:
    """把一条 cue 的颜色归到色板里的某一格。

    彩色优先按色相归（暗粉和粉同色相），色相不沾边才退回 RGB 距离；
    灰阶（白/黑）只在灰阶色板里找，免得白色被吸到粉色上。
    """
    hue, sat = _hue_sat(rgb)
    gray = sat < 0.15
    best_i, best_key = 0, (9, 1e9)
    for i, c in enumerate(centers):
        chue, csat = _hue_sat(c)
        dist = sum((a - b) ** 2 for a, b in zip(rgb, c)) ** 0.5
        if gray or csat < 0.15:
            if gray and csat < 0.15:
                key = (0, dist)
            else:
                key = (2, dist)
        else:
            gap = min(abs(hue - chue), 360 - abs(hue - chue))
            key = (0, gap) if gap <= 18 else (1, dist)
        if key < best_key:
            best_i, best_key = i, key
    return best_i


def stabilize_colors(cues: list[Cue], tolerance: float = 34.0, max_colors: int = 8) -> list[str]:
    """把每条 cue 测到的颜色归并成少数几种（不同说话人 = 不同颜色）。

    压缩/抗锯齿会让同一个颜色测出 ±10 的偏差，这里按 RGB 距离聚类，
    每条 cue 换成所属簇的中心色，输出才稳定。
    """
    raw = [c for c in cues if c.color]
    if not raw:
        return []
    centers: list[tuple[int, int, int]] = []
    counts: list[int] = []
    for c in raw:
        rgb = _rgb_of(c.color)
        for i, center in enumerate(centers):
            if sum((a - b) ** 2 for a, b in zip(rgb, center)) ** 0.5 <= tolerance:
                counts[i] += 1
                break
        else:
            centers.append(rgb)
            counts.append(1)
    # 颜色太多时只留出现最多的几种
    if len(centers) > max_colors:
        keep = sorted(range(len(centers)), key=lambda i: counts[i], reverse=True)[:max_colors]
        centers = [centers[i] for i in keep]
        counts = [counts[i] for i in keep]

    # 少量噪声色（暗场景里粉色测偏了）并到最接近的主色；
    # 真·第三种说话人色通常占比更高、且距离更远，不会被并掉。
    while len(centers) > 1:
        best = None
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])) ** 0.5
                hg = _hue_gap(centers[i], centers[j])
                if not (hg <= 18 or dist < 34):
                    continue
                if min(counts[i], counts[j]) > max(2, len(raw) * 0.06):
                    continue
                score = hg if hg <= 18 else dist
                if best is None or score < best[0]:
                    best = (score, i, j)
        if best is None:
            break
        _, i, j = best
        if counts[i] < counts[j]:
            i, j = j, i
        counts[i] += counts[j]
        del centers[j], counts[j]
    palette = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in centers]
    for c in raw:
        c.color = palette[_assign_index(_rgb_of(c.color), centers)]
    return palette


def run(opts: Options) -> dict[str, Path]:
    t_start = time.time()

    if opts.srt_in:
        _say(f"[1/4] 读取已有日文 SRT: {opts.srt_in}")
        cues = read_srt(opts.srt_in)
        if not cues:
            raise RuntimeError(f"没能从 {opts.srt_in} 解析出任何字幕")
        info = None
        out_dir = opts.out_dir or opts.srt_in.parent
        stem = opts.srt_in.stem
        _say(f"  解析到 {len(cues)} 条")
    else:
        assert opts.video is not None
        _say(f"[1/4] 探测视频: {opts.video}")
        info = probe(opts.video)
        _say(f"  {info}")

        # cue 截图默认落到输出目录下的 cues/，用于对比不同参数的结果
        out_dir = opts.out_dir or opts.video.parent
        stem = opts.video.stem
        if opts.debug_dir is None:
            opts.debug_dir = out_dir / "cues"

        dur = info.duration
        if opts.max_seconds:
            dur = min(dur, opts.max_seconds)
        _say(f"[2/4] 抽帧并检测字幕变化（采样 {opts.sample_fps}fps，字幕带 "
             f"{opts.band[0]:.2f}~{opts.band[1]:.2f}）")

        def gen():
            frames = (stream_band_gray(info, opts.band, fps=opts.sample_fps)
                      if opts.seg.sub_color == "auto"
                      else stream_band_color(info, opts.band, fps=opts.sample_fps))
            for t, g in frames:
                if opts.max_seconds and t > opts.max_seconds:
                    break
                yield t, g

        cues, stats = segment(gen(), opts.seg, fps=opts.sample_fps, total_duration=dur)
        _say(f"  采样 {stats.sampled} 帧，切出 {len(cues)} 条 cue"
             f"（无字幕帧占 {stats.blank_ratio * 100:.0f}%）")
        if not cues:
            raise RuntimeError(
                "一条字幕都没切出来。多半是字幕带选错了，先跑：\n"
                "  jpsub.cmd probe \"你的视频.mp4\"\n"
                "看建议的字幕带区间，再用 --band 0.xx,0.yy 指定。"
            )
        if len(cues) < 3 and stats.blank_ratio > 0.9:
            _say("  ⚠ 切出的字幕很少，字幕带可能偏了，建议先跑 probe 确认")

        _say(f"[3/4] 识别日文（后端 {opts.ocr}）")
        cues = stage_ocr(opts, info, cues)
        before = len(cues)
        cues = merge_repeats(cues)
        if len(cues) != before:
            _say(f"  文本级合并：{before} → {len(cues)} 条")
        kept_shots = _prune_cue_shots(cues, opts.debug_dir)
        if opts.debug_dir:
            _say(f"  cue 截图：保留 {kept_shots} 张（与最终字幕一致）")

    _say(f"[4/4] 翻译并输出")
    stage_translate(opts, cues)

    if opts.max_line:
        for c in cues:
            if c.text:
                c.text = wrap_text(c.text, opts.max_line)

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = stabilize_colors(cues)
    if palette:
        _say(f"  字幕颜色：{', '.join(palette)}")

    written: dict[str, Path] = {}
    written["ass"] = write_ass(cues, out_dir / f"{stem}.ass", mode="target")

    el = time.time() - t_start
    _say("")
    _say(f"完成，共 {len(cues)} 条，用时 {el:.0f}s")
    for k, v in written.items():
        _say(f"  {k:10s} {v}")
    return written

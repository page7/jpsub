"""只抓几帧，快速确认字幕带有没有框住字幕。

不跑变化检测、不跑翻译，只做：
  探测视频 → 按字幕带抓 N 帧 → 存图 → 可选 OCR
再拼成一张 locate_sheet.png，打开就能看定位对不对。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .ocr import make_backend
from .video import grab_band_color, imwrite, probe


def _pbar(done: int, total: int, width: int = 24) -> str:
    frac = done / max(1, total)
    filled = int(width * frac)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total} ({frac * 100:5.1f}%)"


def _pick_times(duration: float, count: int) -> list[float]:
    count = max(1, count)
    if count == 1:
        return [duration * 0.5]
    # 避开片头和片尾，均匀取点
    return [duration * (0.08 + 0.84 * i / (count - 1)) for i in range(count)]


def _make_sheet(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    import cv2

    strip = 26
    gap = 6
    width = max(im.shape[1] for im in images)
    height = sum(im.shape[0] + strip + gap for im in images)
    canvas = np.full((height, width, 3), 245, np.uint8)
    y = 0
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        canvas[y:y + strip, :w] = (40, 40, 40)
        cv2.putText(canvas, label, (8, strip - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += strip
        canvas[y:y + h, :w] = img
        y += h + gap
    return canvas


def _scan_frames(info, band, count: int, log) -> list[tuple[float, np.ndarray]]:
    """先扫一圈，挑墨水最多的几帧，避免抽到没字幕的空画面。"""
    from .cues import SegmentConfig, ink_map

    cfg = SegmentConfig()
    n = min(60, max(24, count * 5))
    log(f"  扫描 {n} 个时间点，挑出有字幕的 {count} 帧…")
    scored: list[tuple[float, float, np.ndarray]] = []
    for t in _pick_times(info.duration, n):
        img = grab_band_color(info, t, band)
        if img is None:
            continue
        ink = float(ink_map(img, cfg).mean())
        scored.append((ink, t, img))
    scored.sort(reverse=True)
    picked = scored[:count]
    picked.sort(key=lambda x: x[1])
    return [(t, img) for _, t, img in picked]


def run_locate(
    video: Path,
    band: tuple[float, float] | str,
    out_dir: Path,
    count: int = 6,
    times: list[float] | None = None,
    ocr: str = "paddle",
    log=print,
    **ocr_kw,
) -> tuple[list[tuple[float, str, Path]], Path | None]:
    info = probe(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清掉上一次的定位图，避免新旧混在一起
    for old in out_dir.glob("locate_*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    sheet_path = out_dir / "locate_sheet.png"
    try:
        sheet_path.unlink(missing_ok=True)
    except OSError:
        pass

    if band == "auto":
        from .band import diagnose, estimate_band

        log("  自动检测字幕带…")
        b0, b1, score = estimate_band(info, log=log)
        band = (b0, b1)
        log(f"  字幕带 → {b0:.2f},{b1:.2f}（{diagnose(score)}）")
    assert not isinstance(band, str)
    log(f"  字幕带 {band[0]:.2f}~{band[1]:.2f}，采样 {len(times) if times else count} 帧")

    backend = None
    if ocr != "none":
        try:
            backend = make_backend(ocr, log=log, **ocr_kw)
        except Exception as e:
            log(f"  ! OCR 后端初始化失败，只保存图片：{e}")

    if times:
        points = [max(0.0, min(info.duration - 0.05, float(t))) for t in times]
        frames = [(t, grab_band_color(info, t, band)) for t in points]
    else:
        frames = _scan_frames(info, band, count, log)

    images: list[np.ndarray] = []
    labels: list[str] = []
    results: list[tuple[float, str, Path]] = []
    t0 = time.time()
    for i, (t, img) in enumerate(frames, 1):
        if img is None:
            log(f"  ! {t:.2f}s 取帧失败")
            continue
        path = out_dir / f"locate_{i:02d}_{t:07.2f}.png"
        if not imwrite(path, img):
            log(f"  ! 写入失败：{path}")
        text = ""
        if backend is not None:
            try:
                text = backend.recognize(img)
            except Exception as e:
                log(f"  ! {t:.2f}s OCR 失败：{e}")
        results.append((t, text, path))
        images.append(img)
        labels.append(f"#{i}  t={t:.2f}s")
        sys.stdout.write(f"\r  {_pbar(i, len(frames))} {t:7.2f}s ")
        sys.stdout.flush()
    sys.stdout.write("\n")

    for t, text, _ in results:
        log(f"  {t:7.2f}s  {text or '（未识别到文字）'}")

    if images:
        if imwrite(sheet_path, _make_sheet(images, labels)):
            log(f"  定位拼图：{sheet_path}")
        else:
            log("  ! 定位拼图写入失败")
    log(f"  定位测试完成，共 {len(results)} 张，用时 {time.time() - t0:.0f}s")
    return results, (sheet_path if images else None)

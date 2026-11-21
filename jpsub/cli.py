"""命令行入口。

用法：
  jpsub.cmd run <视频>            # 主流程：视频 → 中文 ASS
  jpsub.cmd probe <视频>          # 只看视频信息 + 建议字幕带，不跑识别
  jpsub.cmd locate <视频>         # 只抓几帧，确认字幕带有没有框住字幕
  jpsub.cmd srt <日文.srt>        # 已有日文字幕，只翻译，输出中文 ASS
  jpsub.cmd ocr <视频> -t 63.5    # 只对某一时刻做 OCR，用来验证字幕带选得对不对
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cues import SegmentConfig
from .pipeline import Options, run as run_pipeline
from .video import VideoInfo, grab_band_color, imwrite, probe
from .srtio import read_srt, write_ass


def _parse_band(s: str, video: Path | None) -> tuple[float, float] | str:
    s = s.strip().lower()
    if s in ("auto", ""):
        return "auto"
    parts = [p for p in s.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--band 格式应为 0.70,0.99（上下边界，相对画面高度），收到: {s}")
    a, b = float(parts[0]), float(parts[1])
    if not (0 <= a < b <= 1):
        raise SystemExit(f"--band 需要 0 ≤ 上 < 下 ≤ 1，收到: {a}, {b}")
    return (a, b)


def _resolve_band(opts: Options, log=print) -> None:
    if opts.band == "auto":  # type: ignore[comparison-overlap]
        assert opts.video is not None
        from .band import diagnose, estimate_band

        info = probe(opts.video)
        b0, b1, score = estimate_band(info, log=log)
        log(f"  自动检测字幕带 → {b0:.2f},{b1:.2f}（{diagnose(score)}）")
        opts.band = (b0, b1)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--band", default="0.70,0.99",
                   help="字幕带上下边界（相对画面高度），如 0.78,0.97；写 auto 自动猜（默认 0.70,0.99）")
    p.add_argument("--fps", type=float, default=6.0, help="变化检测采样帧率，默认 6（字幕边界精度约 1/fps 秒）")
    p.add_argument("--ocr", choices=["paddle", "vlm"], default="paddle", help="OCR 后端，默认 paddle（本地、最准）")
    p.add_argument("--ocr-model", default="Index-Translate-35B-A3B", help="vlm 后端用的模型 ID")
    p.add_argument("--ocr-base-url", default="https://index-translate.bilibili.com/v1", help="vlm 后端 base_url")
    p.add_argument("--ocr-key", default="", help="vlm 后端的 key（B 站免费端点不需要）")
    p.add_argument("--sim", type=float, default=0.62, help="字幕变化判定阈值，越小越敏感（默认 0.62）")
    p.add_argument("--ink", type=int, default=34, help="笔画二值化阈值，字幕偏暗时调小（默认 34）")
    p.add_argument("--sub-color", choices=["auto", "white", "yellow", "black"], default="auto",
                   help="源视频字幕颜色，默认 auto（不按颜色过滤）；画面亮、背景花时指定 white/yellow 能少误切")
    p.add_argument("--min-dur", type=float, default=0.24, help="最短字幕时长秒，低于此值丢弃（默认 0.24）")
    p.add_argument("--confirm", type=int, default=4,
                   help="连续多少帧判定换句，默认 4（约 0.67s；调大更抗抖动，调小反应更快）")
    p.add_argument("--max-seconds", type=float, default=None, help="只处理前 N 秒，用来试参数")
    p.add_argument("--debug-dir", default=None,
                   help="cue 截图目录，默认 <输出目录>/cues；每条 cue 存一张字幕带截图，便于验收")


def _add_translate(p: argparse.ArgumentParser) -> None:
    p.add_argument("--translate", choices=["bilibili", "openai", "none"], default="bilibili",
                   help="翻译后端，默认 bilibili（免费公开端点，非商用）")
    p.add_argument("--model", default=None, help="翻译模型 ID，bilibili 默认 Index-Translate-35B-A3B")
    p.add_argument("--base-url", default=None, help="openai 后端的 base_url，如 https://api.deepseek.com/v1")
    p.add_argument("--api-key", default=None, help="openai 后端的 key（或用环境变量 OPENAI_API_KEY）")
    p.add_argument("--batch", type=int, default=16, help="每批翻译多少条（默认 16）")
    p.add_argument("--concurrency", type=int, default=2, help="并发批数（默认 2，免费端点别调高）")
    p.add_argument("--retries", type=int, default=3, help="失败重试次数（默认 3）")
    p.add_argument("--max-line", type=int, default=20, help="单行超过多少字自动断行，0 表示不断（默认 20）")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jpsub",
        description="日文字幕视频 → 中文 ASS（抽帧 + OCR + 翻译，本地跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="视频 → 中文 ASS（主流程）")
    r.add_argument("video", type=Path, help="输入视频文件")
    r.add_argument("-o", "--out", type=Path, default=None, help="输出目录（默认与视频同目录）")
    _add_common(r)
    _add_translate(r)

    pr = sub.add_parser("probe", help="看视频信息 + 建议字幕带")
    pr.add_argument("video", type=Path)
    pr.add_argument("--fps", type=float, default=1.0, help="采样帧率，默认 1")

    loc = sub.add_parser("locate", help="只抓几帧，确认字幕带有没有框住字幕")
    loc.add_argument("video", type=Path)
    loc.add_argument("-o", "--out", type=Path, default=None,
                     help="定位图输出目录，默认视频目录/locate")
    loc.add_argument("--band", default="0.70,0.99",
                     help="字幕带，如 0.80,0.94；写 auto 自动猜")
    loc.add_argument("--count", type=int, default=6, help="抓几帧，默认 6")
    loc.add_argument("--times", default="", help="自定义采样秒数，逗号分隔，如 60,120,300")
    loc.add_argument("--ocr", choices=["paddle", "vlm", "none"], default="paddle",
                     help="是否顺便 OCR，默认 paddle；none 只存图")
    loc.add_argument("--ocr-model", default="Index-Translate-35B-A3B")
    loc.add_argument("--ocr-base-url", default="https://index-translate.bilibili.com/v1")
    loc.add_argument("--ocr-key", default="")

    s = sub.add_parser("srt", help="已有日文 SRT，只做翻译，输出中文 ASS")
    s.add_argument("srt", type=Path, help="日文 SRT 文件")
    s.add_argument("-o", "--out", type=Path, default=None, help="输出目录")
    _add_translate(s)

    o = sub.add_parser("ocr", help="对指定时刻做一次 OCR，验证字幕带")
    o.add_argument("video", type=Path)
    o.add_argument("-t", "--time", type=float, required=True, help="秒")
    o.add_argument("--save", type=Path, default=None, help="把截出的字幕带存成图片")
    _add_common(o)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"run", "probe", "locate", "srt", "ocr"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv.insert(0, "run")
    elif not argv:
        build_parser().print_help()
        return 1

    args = build_parser().parse_args(argv)

    if args.cmd == "probe":
        info = probe(args.video)
        print(f"文件    {info.path}")
        print(f"分辨率  {info.width} x {info.height}")
        print(f"帧率    {info.fps:.3f}")
        print(f"时长    {info.duration:.1f}s  ({info.duration / 60:.1f} 分钟)")
        print(f"帧数    {info.n_frames}")
        from .band import diagnose, estimate_band

        print("\n扫描字幕位置…")
        b0, b1, score = estimate_band(info, fps=args.fps)
        print(f"\n建议字幕带  --band {b0:.2f},{b1:.2f}")
        print(f"  （画面高 {info.height}px，对应 y={int(info.height * b0)}"
              f"~{int(info.height * b1)}px，高 {int(info.height * (b1 - b0))}px）")
        print(f"  可信度  {score:.3f} — {diagnose(score)}")
        print("\n先确认一下：")
        print(f"  jpsub.cmd ocr \"{args.video}\" -t 60 --band {b0:.2f},{b1:.2f} --save band.png")
        print("  打开 band.png 看字幕是否被完整框住、且没带上别的画面")
        return 0

    if args.cmd == "locate":
        band = _parse_band(args.band, args.video)
        out_dir = args.out or (args.video.parent / "locate")
        times = None
        if args.times.strip():
            try:
                times = [float(x) for x in args.times.replace("，", ",").split(",") if x.strip()]
            except ValueError:
                raise SystemExit(f"--times 格式应为 60,120,300，收到: {args.times}") from None
        from .locate import run_locate

        run_locate(args.video, band, out_dir, count=args.count, times=times,
                   ocr=args.ocr, log=print, model=args.ocr_model,
                   base_url=args.ocr_base_url, api_key=args.ocr_key)
        return 0

    if args.cmd == "ocr":
        info = probe(args.video)
        band = _parse_band(args.band, args.video)
        if band == "auto":
            from .band import diagnose, estimate_band

            b0, b1, score = estimate_band(info)
            print(f"自动检测字幕带 → {b0:.2f},{b1:.2f}（{diagnose(score)}）")
            band = (b0, b1)
        print(f"{info}")
        print(f"字幕带 {band}  时刻 {args.time}s")
        img = grab_band_color(info, args.time, band)  # type: ignore[arg-type]
        if img is None:
            print("取帧失败")
            return 1
        print(f"截图尺寸 {img.shape[1]}x{img.shape[0]}")
        if args.save:
            if imwrite(args.save, img):
                print(f"已保存 {args.save}")
            else:
                print(f"保存失败：{args.save}")
                return 1
        from .ocr import make_backend

        be = make_backend(args.ocr, model=args.ocr_model, base_url=args.ocr_base_url,
                          api_key=args.ocr_key)
        text = be.recognize(img)
        print("\n识别结果:")
        print(text if text else "（空）")
        return 0

    if args.cmd == "srt":
        cues = read_srt(args.srt)
        print(f"读入 {len(cues)} 条")
        from .translate import make_translator

        tr = make_translator(args.translate, model=args.model, api_key=args.api_key,
                             base_url=args.base_url, batch_size=args.batch,
                             concurrency=args.concurrency, retries=args.retries)
        texts = [c.source.replace("\n", " ").strip() for c in cues]
        outs = tr.run(texts)
        for c, o in zip(cues, outs):
            c.text = (o or "").strip()
        out_dir = args.out or args.srt.parent
        stem = args.srt.stem
        print(write_ass(cues, Path(out_dir) / f"{stem}.ass", "target"))
        return 0

    # run
    band = _parse_band(args.band, args.video)
    opts = Options(
        video=args.video,
        out_dir=args.out,
        band=band,  # type: ignore[arg-type]
        sample_fps=args.fps,
        seg=SegmentConfig(sim_thresh=args.sim, ink_thresh=args.ink,
                          min_duration=args.min_dur, sub_color=args.sub_color,
                          confirm_frames=args.confirm),
        ocr=args.ocr,
        ocr_model=args.ocr_model,
        ocr_base_url=args.ocr_base_url,
        ocr_key=args.ocr_key,
        translator=args.translate,
        tr_model=args.model,
        tr_base_url=args.base_url,
        tr_key=args.api_key,
        batch_size=args.batch,
        concurrency=args.concurrency,
        retries=args.retries,
        max_seconds=args.max_seconds,
        max_line=args.max_line,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
    )
    if opts.band == "auto":  # type: ignore[comparison-overlap]
        from .band import diagnose, estimate_band

        b0, b1, score = estimate_band(probe(args.video))
        print(f"自动检测字幕带 → {b0:.2f},{b1:.2f}（{diagnose(score)}）")
        opts.band = (b0, b1)

    run_pipeline(opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

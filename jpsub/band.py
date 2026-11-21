"""字幕带自动定位：猜出字幕大概在画面哪个高度区间。

只给建议，不偷偷改流程 —— 猜错一次整片就废了，宁可让用户看一眼再决定。
"""

from __future__ import annotations

import numpy as np

from .cues import SegmentConfig, ink_map
from .video import VideoInfo, stream_band_gray


def estimate_band(
    info: VideoInfo,
    fps: float = 1.0,
    max_frames: int = 40,
    log=print,
) -> tuple[float, float, float]:
    """返回 (y0, y1, 可信度)。可信度是检出区间内的平均墨水占比，越低越可能是瞎猜。"""
    lo = 0.45
    frames = []
    for i, (t, gray) in enumerate(stream_band_gray(info, (lo, 1.0), fps=fps, target_w=640)):
        frames.append(gray)
        if i + 1 >= max_frames:
            break
    if not frames:
        return (0.70, 0.99, 0.0)

    cfg = SegmentConfig()
    h = frames[0].shape[0]
    acc = np.zeros(h, dtype=np.float64)
    for g in frames:
        bw = ink_map(g, cfg)
        acc += bw.astype(np.float64).mean(axis=1)

    acc /= len(frames)
    if acc.max() <= 0:
        return (0.70, 0.99, 0.0)

    thr = max(acc.max() * 0.25, 0.002)
    on = acc > thr

    # 把所有连续"有字"行段都找出来
    runs: list[tuple[int, int]] = []
    i = h - 1
    while i >= 0:
        if on[i]:
            j = i
            while j >= 0 and on[j]:
                j -= 1
            runs.append((j + 1, i))
            i = j
        else:
            i -= 1
    if not runs:
        return (0.70, 0.99, 0.0)

    # 字幕是一两行字，高度有上限；整块画面/大面积高对比内容不是字幕。
    # 只在高度合理的段里挑：先取墨水足够浓的，再从里面挑比较高的那一段。
    min_h = max(2, int(h * 0.012))
    max_h = max(4, int(h * 0.22))
    fits = [r for r in runs if min_h <= r[1] - r[0] + 1 <= max_h]
    cands = fits or runs
    top = max(float(acc[r[0]:r[1] + 1].mean()) for r in cands)
    dense = [r for r in cands if float(acc[r[0]:r[1] + 1].mean()) >= top * 0.6]
    best = max(dense, key=lambda r: (r[1] - r[0] + 1, float(acc[r[0]:r[1] + 1].mean())))

    y0p, y1p = best
    score = float(acc[y0p:y1p + 1].mean())

    pad = max(2, int(h * 0.02))
    y0p = max(0, y0p - pad)
    y1p = min(h - 1, y1p + pad)
    # 换算回全图相对坐标
    span = 1.0 - lo
    b0 = lo + (y0p / h) * span
    b1 = lo + ((y1p + 1) / h) * span
    # 太窄说明只抓到一行，扩到合理范围
    if b1 - b0 < 0.03:
        b0 = max(0.0, b1 - 0.03)
    return (round(b0, 3), round(min(1.0, b1 + 0.01), 3), score)


def diagnose(score: float) -> str:
    """把可信度翻译成人话。字幕行的墨水占比通常明显高于随机画面纹理。"""
    if score >= 0.06:
        return "信号清晰，大概率就是字幕带"
    if score >= 0.025:
        return "有文字信号，但不算强 —— 建议截一帧确认"
    if score > 0.008:
        return "信号很弱，可能只是画面里的高对比内容，别急着信"
    return "基本没检测到文字 —— 这个视频可能压根没有硬字幕"

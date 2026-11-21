"""比较三种掩码在真实字幕切换点上的"掉落幅度"。

v0 auto：现有 top-hat/black-hat
v1 white：加纯白过滤
v2 white+描边：纯白且附近有暗像素（字幕黑描边），亮背景上也能挑出字
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.cues import SegmentConfig, ink_map, similarity
from jpsub.video import probe, stream_band_color

BAND = (0.80, 0.94)
START, END = 240.0, 330.0
# 真值里的切换时刻
BOUNDS = [243.5, 247.0, 249.5, 253.0, 257.5, 262.0, 266.5, 274.0, 278.5,
          281.5, 285.5, 290.5, 294.0, 298.5, 300.5, 305.5, 309.5, 312.5,
          318.0, 321.5]

info = probe(Path("test/test.mp4"))
cfg = SegmentConfig(ink_thresh=50)


def masks(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.kernel)
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    _, base = cv2.threshold(resp, cfg.ink_thresh, 1, cv2.THRESH_BINARY)
    b = img[:, :, 0].astype(np.int16)
    gg = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(b, gg), r)
    mn = np.minimum(np.minimum(b, gg), r)
    white = (mx >= 170) & ((mx - mn) <= 70)
    dark = (mx <= 110).astype(np.uint8)
    near_dark = cv2.dilate(dark, np.ones((7, 7), np.uint8)).astype(bool)
    v1 = base.astype(bool) & white
    v2 = v1 & near_dark
    return [base.astype(bool), v1, v2]


def close(m):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))


frames = [(t, masks(img)) for t, img in stream_band_color(info, BAND, fps=6.0)
          if START <= t <= END]
print(f"帧数 {len(frames)}")

names = ["v0 auto", "v1 white", "v2 white+描边"]
for i, name in enumerate(names):
    intra, drop = [], []
    for t, ms in frames:
        prev = [f for f in frames if abs(f[0] - (t - 1 / 6)) < 0.01]
        if not prev:
            continue
        s = similarity(close(ms[i]), close(prev[0][1][i]))
        if any(abs(t - b) < 0.12 for b in BOUNDS):
            drop.append(s)
        elif not any(abs(t - b) < 0.35 for b in BOUNDS):
            intra.append(s)
    ink = float(np.mean([m[i].mean() for _, m in frames]))
    print(f"{name:14s} 墨水={ink:.3f} 同句相邻 sim 中位={np.median(intra):.3f} "
          f"| 切换点 sim 中位={np.median(drop) if drop else -1:.3f} "
          f"最大={max(drop) if drop else -1:.3f}  (n={len(drop)})")

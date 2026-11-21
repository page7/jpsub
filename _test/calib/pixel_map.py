"""把一行字幕按像素颜色打成字符图，直接看填充色/描边结构。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")


def show(crop, label):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    rows, cols = 22, 118
    print(f"  {label}  {w}x{h}")
    for ry in range(rows):
        line = ""
        for rx in range(cols):
            blk = hsv[ry * h // rows:(ry + 1) * h // rows,
                      rx * w // cols:(rx + 1) * w // cols]
            hh = float(np.median(blk[:, :, 0]))
            s = float(np.median(blk[:, :, 1]))
            v = float(np.median(blk[:, :, 2]))
            if s >= 60 and 135 <= hh <= 205 and v >= 60:
                ch = "P"  # 粉
            elif v >= 195 and s <= 45:
                ch = "W"  # 白
            elif v <= 75:
                ch = "K"  # 黑
            else:
                ch = "."
            line += ch
        print("   |" + line + "|")


for t in (30, 270, 1500):
    img = grab_band_color(info, t, (0.80, 0.94))
    print(f"--- t={t}s")
    for page in be.ocr.predict(img):
        ts = page.get("rec_texts") or []
        ps = page.get("rec_polys") or []
        for i, txt in enumerate(ts):
            if not txt.strip() or i >= len(ps):
                continue
            p = np.asarray(ps[i])
            x0, x1 = int(p[:, 0].min()), int(p[:, 0].max())
            y0, y1 = int(p[:, 1].min()), int(p[:, 1].max())
            crop = img[max(0, y0):y1, max(0, x0):min(img.shape[1], x1)]
            if crop.size:
                # 只取左边 1/4，字大一点看得清
                show(crop[:, :max(20, crop.shape[1] // 4)], f"「{txt[:22]}」")
            break
        break

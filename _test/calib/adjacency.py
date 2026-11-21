"""验证：真填充色紧挨着白描边，背景透色不挨着。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")

for t in (30, 48.43, 61.77, 91.43, 270.0, 112.0, 1500.0):
    img = grab_band_color(info, t, (0.805, 0.890))
    be.recognize(img)
    print(f"--- t={t}")
    for poly in getattr(be, "last_polys", []):
        p = np.asarray(poly)
        x0, y0 = max(0, int(p[:, 0].min())), max(0, int(p[:, 1].min()))
        x1, y1 = int(p[:, 0].max()), int(p[:, 1].max())
        crop = img[y0:y1, x0:x1]
        if crop.size < 2000:
            continue
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                       cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
        mask = resp > 40
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(int)
        s = hsv[:, :, 1].astype(int)
        white = mask & (v >= 190) & (s <= 50)
        black = mask & (v <= 80)
        chroma = mask & (s >= 70) & (v >= 90)
        wk = np.ones((5, 5), np.uint8)
        near_white = cv2.dilate(white.astype(np.uint8), wk) > 0
        near_black = cv2.dilate(black.astype(np.uint8), wk) > 0
        n = max(1, int(chroma.sum()))
        pix = crop[chroma]
        if len(pix):
            b, gg, r = (int(np.median(pix[:, i])) for i in range(3))
            hexv = f"#{r:02X}{gg:02X}{b:02X}"
        else:
            hexv = "-"
        print(f"   框 {crop.shape[1]}x{crop.shape[0]} 总数={int(mask.sum())} "
              f"白={int(white.sum())} 黑={int(black.sum())} 彩={n} {hexv}")
        print(f"      彩挨白={float((chroma & near_white).sum()) / n:.2f} "
              f"彩挨黑={float((chroma & near_black).sum()) / n:.2f}")

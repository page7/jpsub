import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import detect_text_color, make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
BAND = (0.80, 0.94)

for t in (963.1, 1182.1, 657.8, 1165.2, 1766.8, 1245.4):
    img = grab_band_color(info, t, BAND)
    txt = be.recognize(img)
    print(f"--- t={t} 「{txt[:26]}」 -> {detect_text_color(img, be.last_polys)}")
    for poly in be.last_polys:
        p = np.asarray(poly)
        x0, y0 = max(0, int(p[:, 0].min())), max(0, int(p[:, 1].min()))
        x1, y1 = int(p[:, 0].max()), int(p[:, 1].max())
        crop = img[y0:y1, x0:x1]
        if crop.size < 1500:
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
        near = cv2.dilate((white | black).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        n = int(mask.sum())
        nc = int(chroma.sum())
        if nc:
            pix = crop[chroma]
            b, gg, r = (int(np.median(pix[:, i])) for i in range(3))
            adj = float((chroma & near).sum()) / nc
            print(f"   n={n} 白={int(white.sum())} 黑={int(black.sum())} 彩={nc} "
                  f"#{r:02X}{gg:02X}{b:02X} 挨边={adj:.2f}")

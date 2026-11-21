import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import detect_text_color, make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
for t in (48.43, 91.43, 61.77, 112.0, 270.0):
    img = grab_band_color(info, t, (0.805, 0.890))
    txt = be.recognize(img)
    print(f"--- t={t} {txt[:26]}")
    print("   with polys:", detect_text_color(img, getattr(be, "last_polys", None)))
    print("   no polys  :", detect_text_color(img, None))
    # 背景中位色（掩码外）
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    bg = img[resp <= 40]
    if len(bg):
        b, gg, r = (int(np.median(bg[:, i])) for i in range(3))
        print(f"   背景中位色 #{r:02X}{gg:02X}{b:02X}")

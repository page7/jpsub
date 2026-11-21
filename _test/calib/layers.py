"""判断哪一层是"填充"：离背景远的那一层。描边贴着背景，填充被描边包着。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")


def analyze(img, poly):
    x0, y0, x1, y1 = (int(v) for v in (poly[:, 0].min(), poly[:, 1].min(),
                                       poly[:, 0].max(), poly[:, 1].max()))
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    mask = (resp > 40).astype(np.uint8)
    n = int(mask.sum())
    if n < 100:
        print("    掩码太小")
        return
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(int)
    s = hsv[:, :, 1].astype(int)
    h = hsv[:, :, 0].astype(int)
    # 背景 = 掩码外；ring = 紧贴背景的那一圈
    bg = (mask == 0).astype(np.uint8)
    ring = (cv2.dilate(bg, np.ones((3, 3), np.uint8)) > 0) & (mask > 0)
    groups = {
        "white": (v >= 190) & (s <= 50),
        "black": v <= 80,
        "chroma": (s >= 70) & (v >= 90),
    }
    print(f"    掩码 {n}px")
    for name, gm in groups.items():
        gm = gm & (mask > 0)
        cnt = int(gm.sum())
        if not cnt:
            continue
        adj = float((gm & ring).sum()) / cnt
        pix = crop[gm]
        b, gg, r = (int(np.median(pix[:, i])) for i in range(3))
        print(f"      {name:6s} {cnt:5d}px ({cnt / n:5.1%}) 贴边率={adj:5.2f} "
              f"中位色 #{r:02X}{gg:02X}{b:02X}")


for t in (30, 60, 120, 270, 300, 600, 1000, 1500, 2000, 2500, 3000):
    img = grab_band_color(info, t, (0.80, 0.94))
    print(f"--- t={t}s")
    for page in be.ocr.predict(img):
        ts = page.get("rec_texts") or []
        ps = page.get("rec_polys") or []
        for i, txt in enumerate(ts):
            if not txt.strip() or i >= len(ps):
                continue
            print(f"  「{txt[:24]}」")
            analyze(img, np.asarray(ps[i]))
            break
        break

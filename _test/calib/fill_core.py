"""用距离变换取笔画最里面那层像素，测真正的填充色（排除白色描边）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")


def fill_color(img, poly):
    x0, y0, x1, y1 = (int(v) for v in (poly[:, 0].min(), poly[:, 1].min(),
                                       poly[:, 0].max(), poly[:, 1].max()))
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    mask = (resp > 40).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return None
    for frac in (0.55, 0.7):
        core = dist >= max(1.1, dist.max() * frac)
        pix = crop[core]
        if len(pix) < 20:
            continue
        b, gg, r = (int(np.median(pix[:, i])) for i in range(3))
        hsv = cv2.cvtColor(np.uint8([[[b, gg, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        print(f"    core{frac}: n={len(pix):4d} #{r:02X}{gg:02X}{b:02X} "
              f"H={hsv[0]} S={hsv[1]} V={hsv[2]}")
        return (r, gg, b)
    return None


for t in (30, 60, 120, 270, 300, 600, 1000, 1500, 2000, 2500, 3000):
    img = grab_band_color(info, t, (0.80, 0.94))
    if img is None:
        continue
    print(f"--- t={t}s")
    for page in be.ocr.predict(img):
        ts = page.get("rec_texts") or []
        ps = page.get("rec_polys") or []
        for i, txt in enumerate(ts):
            if txt.strip() and i < len(ps):
                print(f"  「{txt[:24]}」")
                fill_color(img, np.asarray(ps[i]))

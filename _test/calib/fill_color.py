"""看字幕的填充色：在 OCR 框内，用 描边+填充 掩码，腐蚀掉描边后取中位色。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")


def fill_colors(img, poly):
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
    x0, y0, x1, y1 = (int(v) for v in (poly[:, 0].min(), poly[:, 1].min(),
                                       poly[:, 0].max(), poly[:, 1].max()))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    mask = (resp > 40).astype(np.uint8)
    er = cv2.erode(mask, np.ones((3, 3), np.uint8))
    for name, mm in (("原掩码", mask), ("腐蚀后", er)):
        pix = crop[mm > 0]
        if len(pix) < 20:
            print(f"   {name}: 像素太少")
            continue
        b, g2, r = (int(np.median(pix[:, i])) for i in range(3))
        cv2.cvtColor(np.uint8([[[b, g2, r]]]), cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(np.uint8([[[b, g2, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        print(f"   {name}: n={len(pix):5d} #{r:02X}{g2:02X}{b:02X} "
              f"H={hsv[0]} S={hsv[1]} V={hsv[2]}")
    # 最常见颜色（量化后）
    q = (crop[mask > 0] // 32 * 32)
    if len(q):
        u, c = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
        top = sorted(zip(c, u.tolist()), reverse=True)[:5]
        cols = " ".join(f"#{r:02X}{g2:02X}{b:02X}x{n}" for n, (b, g2, r) in top)
        print(f"   顶部色: {cols}")


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
                print(f"  文字「{txt[:26]}」")
                fill_colors(img, np.asarray(ps[i]))

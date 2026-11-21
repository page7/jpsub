"""抽 30 个点，看这片的字幕都是什么颜色。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
cnt = Counter()
step = info.duration / 31
for i in range(1, 31):
    t = i * step
    img = grab_band_color(info, t, (0.80, 0.94))
    if img is None:
        continue
    polys = []
    for page in be.ocr.predict(img):
        ps = page.get("rec_polys") or []
        ts = page.get("rec_texts") or []
        for j, txt in enumerate(ts):
            if txt.strip() and j < len(ps):
                polys.append((np.asarray(ps[j]), txt))
    if not polys:
        continue
    # 每行文字：亮像素的中位色
    line_colors = []
    for poly, txt in polys:
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        pix = img[m > 0]
        if len(pix) < 80:
            continue
        hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        v = hsv[:, 2].astype(int)
        sel = v >= np.percentile(v, 75)
        if sel.sum() < 40:
            continue
        b, g, r = (int(np.median(pix[sel][:, k])) for k in range(3))
        line_colors.append((f"#{r:02X}{g:02X}{b:02X}", txt[:22], b, g, r))
    if not line_colors:
        continue
    hexv, txt, b, g, r = line_colors[0]
    cnt[hexv] += 1
    print(f"t={t:7.1f}s {hexv} (B{b} G{g} R{r}) {txt}")
print("\n颜色分布：", dict(cnt))

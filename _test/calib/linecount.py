"""统计全片字幕占几行：抽 24 个时刻做全画面 OCR，记录字幕框的 y 范围。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
step = info.duration / 25
deep = Counter()
rows = []
for i in range(1, 25):
    t = i * step
    img = grab_band_color(info, t, (0.0, 1.0), target_w=1280)
    if img is None:
        continue
    subs = []
    for page in be.ocr.predict(img):
        ts = page.get("rec_texts") or []
        ps = page.get("rec_polys") or []
        for j, txt in enumerate(ts):
            p = np.asarray(ps[j])
            y0 = p[:, 1].min() / img.shape[0]
            y1 = p[:, 1].max() / img.shape[0]
            if y0 > 0.70 and txt.strip():  # 只看下部字幕
                subs.append((y0, y1, txt))
    subs.sort()
    if not subs:
        rows.append((t, 0, None, None, ""))
        continue
    lines = len(subs)
    deep[lines] += 1
    rows.append((t, lines, subs[0][0], subs[-1][1], subs[0][2]))
    print(f"t={t:7.1f}s 行数={lines} y={subs[0][0]:.3f}-{subs[-1][1]:.3f} {subs[0][2][:40]}")
print("\n行数分布：", dict(deep))
print(f"最下面一条框到 y={max((r[3] or 0) for r in rows):.3f}")
print(f"两行字幕占比 {deep.get(2, 0)}/{sum(deep.values())}")

"""验证填充色识别：黑字/粉字/白字。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jpsub.ocr import detect_text_color, make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
cnt = Counter()
for t in (30, 60, 120, 200, 240, 270, 300, 450, 600, 900, 1000, 1200,
          1500, 1800, 2000, 2200, 2500, 2800, 3000, 3150):
    img = grab_band_color(info, t, (0.80, 0.94))
    if img is None:
        continue
    text = be.recognize(img)
    color = detect_text_color(img, getattr(be, "last_polys", None))
    cnt[color] += 1
    print(f"t={t:5d}s {color:9s} {text[:34]}")
print("\n分布:", dict(cnt))

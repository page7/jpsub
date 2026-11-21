import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
for t in (270, 278, 285, 295, 305, 312):
    img = grab_band_color(info, t, (0.0, 1.0), target_w=1280)
    print("---", t)
    for page in be.ocr.predict(img):
        ts = page.get("rec_texts") or []
        ps = page.get("rec_polys") or []
        for i, txt in enumerate(ts):
            p = np.asarray(ps[i])
            print(f"  y={p[:, 1].min() / img.shape[0]:.3f}-{p[:, 1].max() / img.shape[0]:.3f} "
                  f"x={p[:, 0].min() / img.shape[1]:.2f}-{p[:, 0].max() / img.shape[1]:.2f} {txt}")

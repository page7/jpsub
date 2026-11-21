import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jpsub.pipeline import _hue_gap, _hue_sat, _rgb_of

for hexv in ("#A55891", "#000000", "#555C99", "#763F67", "#FFFFFF"):
    rgb = _rgb_of(hexv)
    print(hexv, rgb, "H/S =", tuple(round(x, 1) for x in _hue_sat(rgb)))

pairs = [("#A55891", "#763F67"), ("#555C99", "#763F67"),
         ("#A55891", "#555C99"), ("#FFFFFF", "#A55891")]
for a, b in pairs:
    ra, rb = _rgb_of(a), _rgb_of(b)
    dist = sum((x - y) ** 2 for x, y in zip(ra, rb)) ** 0.5
    print(f"{a} vs {b}: dist={dist:.1f} huegap={_hue_gap(ra, rb):.1f}")

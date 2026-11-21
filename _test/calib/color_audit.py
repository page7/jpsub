"""全片颜色体检：色板分布 + 抽样回测（重新识别一次，看和存下来的一致不）。"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jpsub.ocr import detect_text_color, make_backend
from jpsub.video import grab_band_color, probe

VIDEO = Path("test/test.mp4")
BAND = (0.80, 0.94)
cues = json.loads(Path(sys.argv[1]).read_text("utf-8"))

cnt = Counter(c["color"] for c in cues)
dur = Counter()
for c in cues:
    dur[c["color"]] += c["end"] - c["start"]
print(f"共 {len(cues)} 条")
for color, n in cnt.most_common():
    print(f"  {color or '(空)':9s} {n:4d} 条  {dur[color] / 60:5.1f} 分钟")

# 每个颜色的示例
print("\n每个颜色举 3 条：")
shown = Counter()
for c in cues:
    if shown[c["color"]] < 3:
        shown[c["color"]] += 1
        print(f"  {c['color'] or '(空)':9s} {c['start']:7.1f}s {c['ja'][:30]}")

# 抽样回测
info = probe(VIDEO)
be = make_backend("paddle")
def _rgb(hexv: str):
    h = hexv.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def same(a: str, b: str, tol: float = 24) -> bool:
    if not a or not b:
        return a == b
    return sum((x - y) ** 2 for x, y in zip(_rgb(a), _rgb(b))) ** 0.5 <= tol


step = max(1, len(cues) // 30)
picks = cues[::step]
ok = 0
bad = []
for c in picks:
    t = c["start"] + min(0.4, (c["end"] - c["start"]) * 0.4)
    img = grab_band_color(info, t, BAND)
    if img is None:
        continue
    be.recognize(img)
    got = detect_text_color(img, getattr(be, "last_polys", None))
    if same(got, c["color"]):
        ok += 1
    else:
        bad.append((c["start"], c["color"], got, c["ja"][:24]))
print(f"\n抽样回测 {len(picks)} 条：一致 {ok}，不一致 {len(bad)} "
      f"({ok / max(1, len(picks)):.0%})")
for b in bad[:12]:
    print(f"  {b[0]:7.1f}s 存的={b[1]} 重测={b[2]} {b[3]}")

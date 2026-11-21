"""核对少数派颜色：这些 cue 的取样帧再测一次，看是不是稳定存在的第三种颜色。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jpsub.ocr import detect_text_color, make_backend
from jpsub.video import grab_band_color, probe

info = probe(Path("test/test.mp4"))
be = make_backend("paddle")
BAND = (0.80, 0.94)
cues = json.loads(Path("_test/out_full_color/test.cues.json").read_text("utf-8"))

for target in ("#555C99", "#763F67"):
    rows = [c for c in cues if c["color"] == target]
    print(f"=== {target} 共 {len(rows)} 条")
    for c in rows[:8]:
        times = [c["start"] + 0.35, (c["start"] + c["end"]) / 2]
        got = []
        for t in times:
            img = grab_band_color(info, t, BAND)
            if img is None:
                got.append("-")
                continue
            txt = be.recognize(img)
            got.append(f"{detect_text_color(img, be.last_polys)}「{txt[:12]}」")
        print(f"  {c['start']:7.1f}-{c['end']:7.1f} {' | '.join(got)}")

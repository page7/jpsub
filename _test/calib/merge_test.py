import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jpsub.pipeline import stabilize_colors
from jpsub.srtio import Cue

raw = (["#A55891"] * 284 + ["#000000"] * 134 + ["#555C99"] * 20
       + ["#763F67"] * 5 + ["#FFFFFF"] * 1)
cues = [Cue(start=i, end=i + 1, color=c) for i, c in enumerate(raw)]
palette = stabilize_colors(cues)
from collections import Counter

print("色板:", palette)
print("归并后:", Counter(c.color for c in cues))
print("暗粉那 5 条 ->", {c.color for c in cues[438:443]})

import json
import sys
from pathlib import Path

cues = json.loads(Path(sys.argv[1]).read_text("utf-8"))
a, b = float(sys.argv[2]), float(sys.argv[3])
for c in cues:
    if a <= c["start"] <= b:
        d = c["end"] - c["start"]
        print(f"{c['start']:7.2f}-{c['end']:7.2f} ({d:5.1f}s) {c['ja'][:70]}")

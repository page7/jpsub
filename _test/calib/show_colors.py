import json
import sys
from collections import Counter
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text("utf-8"))
print(Counter(c["color"] for c in d))
for c in d:
    print(f"{c['start']:7.1f} {c['color']:9s} {c['ja'][:34]}")

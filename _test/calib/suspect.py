"""廉价可疑度：时长长、但文本很短 => 很可能把多句并成了一条。"""
import json
import re
import sys
from pathlib import Path

cues = json.loads(Path(sys.argv[1]).read_text("utf-8"))
H = int(sys.argv[2]) if len(sys.argv) > 2 else -1


def chars(s: str) -> int:
    return len(re.sub(r"[\s「」『』()（）…、。,.!?！？~～ー♥❤ッっ]", "", s))


rows = []
for c in cues:
    d = c["end"] - c["start"]
    n = chars(c["ja"])
    rows.append((c["start"], d, n, d / max(1, n), c["ja"]))

print("时长>6s 且 每秒字数<0.35 的条（越极端越像吞句）：")
sus = [r for r in rows if r[1] > 6 and r[3] > 0.35]
for s, d, n, r, t in sus:
    print(f"  {s:7.1f}s 时长{d:5.1f}s 字{n:3d} {r:4.2f}s/字  {t[:34]}")
print(f"\n共 {len(sus)}/{len(rows)} 条（{len(sus) / len(rows):.1%}）")

print("\n按分钟统计可疑条：")
per = {}
for s, d, n, r, t in rows:
    m = int(s // 60)
    st = per.setdefault(m, [0, 0])
    st[0] += 1
    st[1] += (d > 6 and r > 0.35)
for m in sorted(per):
    n, bad = per[m]
    print(f"  {m:2d}分: {bad}/{n}" + ("  <-- 全片最差" if n and bad / n > 0.5 else ""))

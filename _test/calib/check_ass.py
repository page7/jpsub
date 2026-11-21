"""检查 ASS：字段数、时间码、文本里的非法字符，找出让 libass 罢工的那一行。"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text("utf-8-sig")
print(f"文件 {path}  {len(raw)} 字节  行尾 CRLF={raw.count(chr(13))}")
lines = raw.splitlines()

TS = re.compile(r"^\d+:\d{2}:\d{2}\.\d{2}$")
in_events = False
bad = []
n = 0
prev_start = -1
for i, line in enumerate(lines, 1):
    if line.strip() == "[Events]":
        in_events = True
        continue
    if not in_events or not line.strip():
        continue
    if line.startswith("Format:"):
        continue
    n += 1
    if not line.startswith("Dialogue:"):
        bad.append((i, "不是 Dialogue 行", line[:80]))
        continue
    fields = line.split(",", 9)
    if len(fields) != 10:
        bad.append((i, f"字段数 {len(fields)}", line[:80]))
        continue
    _, st, et, style, name, ml, mr, mv, eff, text = fields
    if not TS.match(st) or not TS.match(et):
        bad.append((i, f"时间码 {st} → {et}", line[:80]))
    # 控制字符 / 未转义的换行 / 大括号
    for ch in text:
        if ord(ch) < 32:
            bad.append((i, f"控制字符 U+{ord(ch):04X}", repr(text[:60])))
            break
    if text.count("{") != text.count("}"):
        bad.append((i, "花括号不配对", text[:80]))
    # 起始时间是否递增
    m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", st)
    if m:
        sec = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 100
        if sec < prev_start:
            bad.append((i, f"时间倒序 {prev_start:.2f} → {sec:.2f}", text[:40]))
        prev_start = sec

print(f"事件行 {n} 条，问题 {len(bad)} 处")
for b in bad[:40]:
    print(f"  第 {b[0]} 行: {b[1]}  {b[2]}")

# 看看 60~90 秒附近的原文
lo, hi = 55, 95
print(f"\n{lo}~{hi} 秒的事件：")
for line in lines:
    if not line.startswith("Dialogue:"):
        continue
    f = line.split(",", 9)
    if len(f) < 10:
        continue
    m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", f[1])
    sec = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 100
    if lo <= sec <= hi:
        print(f"  {f[1]}-{f[2]}  {repr(f[9][:70])}")

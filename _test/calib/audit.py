"""全片体检：看条数/时长分布/可疑 cue，并可用 OCR 抽查长条是否吞了多句。

用法：
  python _test/calib/audit.py _test/out_full/test.cues.json
  python _test/calib/audit.py _test/out_full/test.cues.json --ocr
"""
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VIDEO = Path("test/test.mp4")
BAND = (0.805, 0.890)
DUR = 3213.8


def norm(s: str) -> str:
    return "".join(s.split())


def main() -> None:
    path = Path(sys.argv[1])
    do_ocr = "--ocr" in sys.argv
    cues = json.loads(path.read_text("utf-8"))
    n = len(cues)
    durs = [c["end"] - c["start"] for c in cues]
    cover = sum(durs) / DUR
    durs_sorted = sorted(durs)
    print(f"总条数 {n}  覆盖率 {cover:.1%}  中位时长 {durs_sorted[n // 2]:.2f}s  "
          f"p90 {durs_sorted[int(n * 0.9)]:.2f}s  最长 {max(durs):.1f}s")
    print(f"空日文 {sum(1 for c in cues if not c['ja'].strip())} 条  "
          f"空译文 {sum(1 for c in cues if not c['zh'].strip())} 条")

    buckets = Counter()
    for d in durs:
        if d < 0.8:
            buckets["<0.8s 碎片"] += 1
        elif d <= 6:
            buckets["0.8-6s 正常"] += 1
        elif d <= 10:
            buckets["6-10s 偏长"] += 1
        else:
            buckets[">10s 疑似吞句"] += 1
    print("\n时长分布：")
    for k in ("<0.8s 碎片", "0.8-6s 正常", "6-10s 偏长", ">10s 疑似吞句"):
        print(f"  {k:14s} {buckets[k]:4d}")

    # 相邻同文（重复切）
    dup = 0
    for a, b in zip(cues, cues[1:]):
        if norm(a["ja"]) and SequenceMatcher(None, norm(a["ja"]), norm(b["ja"])).ratio() > 0.9:
            dup += 1
    print(f"相邻同文（重复切） {dup} 对")

    # 空白间隔
    gaps = []
    prev_end = 0.0
    for c in cues:
        if c["start"] - prev_end > 4:
            gaps.append((prev_end, c["start"]))
        prev_end = max(prev_end, c["end"])
    if DUR - prev_end > 4:
        gaps.append((prev_end, DUR))
    print(f"≥4s 无字幕间隔 {len(gaps)} 段，共 {sum(b - a for a, b in gaps) / 60:.1f} 分钟")
    if gaps:
        print("  最长 5 段：", " ".join(f"{a:.0f}-{b:.0f}s" for a, b in
                                     sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)[:5]))

    print("\n按分钟分布（条数/最长/碎片数）：")
    per = {}
    for c in cues:
        m = int(c["start"] // 60)
        st = per.setdefault(m, {"n": 0, "max": 0.0, "short": 0, "long": 0})
        d = c["end"] - c["start"]
        st["n"] += 1
        st["max"] = max(st["max"], d)
        st["short"] += d < 0.8
        st["long"] += d > 6
    for m in range(int(DUR // 60) + 1):
        st = per.get(m)
        if not st:
            print(f"  {m:2d}分: —— 无字幕")
            continue
        flag = ""
        if st["long"]:
            flag += f" ⚠{st['long']}条>6s"
        if st["short"] > st["n"] * 0.3:
            flag += f" ⚠碎片{st['short']}/{st['n']}"
        print(f"  {m:2d}分: {st['n']:3d}条  最长{st['max']:5.1f}s{flag}")

    sus = [c for c in cues if c["end"] - c["start"] > 6]
    print(f"\n>6s 的 cue：{len(sus)} 条")
    for c in sus:
        print(f"  {c['start']:7.2f}-{c['end']:7.2f} ({c['end'] - c['start']:5.1f}s) {c['ja'][:36]}")

    if do_ocr and sus:
        from jpsub.ocr import make_backend
        from jpsub.video import grab_band_color, probe
        info = probe(VIDEO)
        be = make_backend("paddle")
        print("\n长条抽查（首/中/尾三处 OCR 是否同文）：")
        bad = 0
        for c in sus:
            ts = [c["start"] + 0.4, (c["start"] + c["end"]) / 2, c["end"] - 0.4]
            texts = []
            for t in ts:
                img = grab_band_color(info, t, BAND)
                texts.append(be.recognize(img) if img is not None else "")
            r1 = SequenceMatcher(None, norm(texts[0]), norm(texts[1])).ratio()
            r2 = SequenceMatcher(None, norm(texts[1]), norm(texts[2])).ratio()
            ok = r1 > 0.6 and r2 > 0.6
            if not ok:
                bad += 1
            print(f"  {c['start']:7.1f}s {'OK ' if ok else '✗吞'} "
                  f"| {texts[0][:18]} | {texts[1][:18]} | {texts[2][:18]}")
        print(f"  长条里疑似吞句 {bad}/{len(sus)} 条")


if __name__ == "__main__":
    main()

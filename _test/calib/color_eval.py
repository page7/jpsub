"""按颜色过滤前后对比：用 0.5s 逐帧 OCR 当真值，给 segment() 打分。

用法：python _test/calib/color_eval.py white
     python _test/calib/color_eval.py auto
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jpsub.cues import SegmentConfig, segment
from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe, stream_band_color, stream_band_gray

VIDEO = Path("test/test.mp4")
BAND = (0.80, 0.94)
START, END, STEP = 240.0, 330.0, 0.5
CACHE = Path(__file__).with_name("truth_240_330.json")

info = probe(VIDEO)
cache = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}
be = None
times = [round(START + i * STEP, 2) for i in range(int((END - START) / STEP) + 1)]
for t in times:
    k = f"{t}"
    if k in cache:
        continue
    if be is None:
        be = make_backend("paddle")
    img = grab_band_color(info, t, BAND)
    cache[k] = be.recognize(img) if img is not None else ""
    if len(cache) % 20 == 0:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
CACHE.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")

seq = [(t, cache[f"{t}"]) for t in times]
runs = []
for t, txt in seq:
    if runs and SequenceMatcher(None, runs[-1]["text"], txt).ratio() > 0.86:
        runs[-1]["end"] = t
        if len(txt) > len(runs[-1]["text"]):
            runs[-1]["text"] = txt
    else:
        runs.append({"start": t, "end": t, "text": txt})
runs = [r for r in runs if r["end"] - r["start"] >= 0.4]
real = [r for r in runs if r["text"].strip()]

print(f"=== {START}-{END}s 真值 {len(real)} 条字幕（{len(runs) - len(real)} 段空白）")
for r in real:
    print(f"  {r['start']:7.1f}-{r['end']:7.1f} {r['text'][:46]}")


def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def score(cues):
    tp = fp = 0
    for c in cues:
        best = max(real, key=lambda r: ov(c.start, c.end, r["start"], r["end"]), default=None)
        if best and ov(c.start, c.end, best["start"], best["end"]) / max(0.01, c.duration) >= 0.6:
            tp += 1
        else:
            fp += 1
    hit = sum(1 for r in real if any(
        ov(c.start, c.end, r["start"], r["end"]) / max(0.01, r["end"] - r["start"]) >= 0.7
        for c in cues))
    p = tp / max(1, tp + fp)
    rc = hit / len(real)
    return p, rc, 2 * p * rc / max(1e-6, p + rc), len(cues)


mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
stream = (stream_band_gray(info, BAND, fps=6.0) if mode == "auto"
          else stream_band_color(info, BAND, fps=6.0))
raw = [(t, g) for t, g in stream if START <= t <= END]
print(f"\n=== {mode}: 采样 {len(raw)} 帧")

for ink, sim, md in ((34, 0.85, 0.6), (50, 0.85, 0.6), (34, 0.80, 0.6)):
    cfg = SegmentConfig(ink_thresh=ink, sim_thresh=sim, min_duration=md, sub_color=mode)
    cues, st = segment(list(raw), cfg, fps=6.0, total_duration=END)
    p, rc, f1, n = score(cues)
    print(f"ink={ink} sim={sim:.2f} md={md}: cue={n:3d} P={p:.2f} R={rc:.2f} F1={f1:.2f}")
    if (ink, sim, md) == (50, 0.85, 0.6):
        for c in cues:
            print(f"    {c.start:7.1f}-{c.end:7.1f} ({c.duration:5.1f}s)")

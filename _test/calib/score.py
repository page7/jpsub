"""用逐帧 OCR 的真值给 segment() 打分，网格搜参数。"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jpsub.cues import SegmentConfig, segment
from jpsub.video import probe, stream_band_gray

video = Path(r"D:\ai\jpsub\test\test.mp4")
band = (0.79, 0.91)
END = 120.0
STEP = 0.5
CACHE = Path(__file__).with_name("ocr_cache.json")
key_base = f"{band[0]},{band[1]}"

info = probe(video)
cache = json.loads(CACHE.read_text("utf-8"))
seq = [(round(i * STEP, 2), cache.get(f"{key_base}@{round(i * STEP, 2)}", ""))
       for i in range(int(END / STEP) + 1)]

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

raw = [(t, g) for t, g in stream_band_gray(info, band, fps=6.0) if t <= END]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def score(cues):
    tp = fp = 0
    for c in cues:
        best = max(real, key=lambda r: overlap(c.start, c.end, r["start"], r["end"]),
                   default=None)
        if best and overlap(c.start, c.end, best["start"], best["end"]) / max(0.01, c.duration) >= 0.6:
            tp += 1
        else:
            fp += 1
    hit = sum(1 for r in real if any(
        overlap(c.start, c.end, r["start"], r["end"]) / max(0.01, r["end"] - r["start"]) >= 0.7
        for c in cues))
    return tp, fp, hit


print(f"真值：{len(real)} 条字幕，{len(runs) - len(real)} 段空白")
best = None
for ink in (34, 50):
    for sim in (0.62, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for md in (0.24, 0.40, 0.60):
            for cf in (2, 3):
                cfg = SegmentConfig(ink_thresh=ink, sim_thresh=sim,
                                    min_duration=md, confirm_frames=cf)
                cues, _ = segment(list(raw), cfg, fps=6.0, total_duration=END)
                tp, fp, hit = score(cues)
                rec = hit / len(real)
                prec = tp / max(1, tp + fp)
                f1 = 2 * prec * rec / max(1e-6, prec + rec)
                line = (f"ink={ink:2d} sim={sim:.2f} min_dur={md:.2f} cf={cf}  "
                        f"cue={len(cues):3d} 精确={prec:.2f} 召回={rec:.2f} F1={f1:.2f}")
                if best is None or f1 > best[0]:
                    best = (f1, line)
                print(line)
print("\n最佳:", best[1])

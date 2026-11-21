"""用逐帧 OCR 当"标准答案"，看切出来的 cue 对不对。

每 0.5s 对字幕带做一次 OCR，把文本相同的连续区间合并成 true run，
再和 segment() 切出来的 cue 对比。
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jpsub.cues import SegmentConfig, segment
from jpsub.ocr import make_backend
from jpsub.video import grab_band_color, probe, stream_band_gray

video = Path(r"D:\ai\jpsub\test\test.mp4")
band = (0.79, 0.91)
END = 120.0
STEP = 0.5
CACHE = Path(__file__).with_name("ocr_cache.json")

info = probe(video)
cache = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}
key_base = f"{band[0]},{band[1]}"

be = None
times = [round(i * STEP, 2) for i in range(int(END / STEP) + 1)]
for t in times:
    key = f"{key_base}@{t}"
    if key in cache:
        continue
    if be is None:
        be = make_backend("paddle")
    img = grab_band_color(info, t, band)
    cache[key] = be.recognize(img) if img is not None else ""
    if len(cache) % 10 == 0:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), "utf-8")
CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), "utf-8")

seq = [(t, cache[f"{key_base}@{t}"]) for t in times]

# 合并成文本 run（容忍 OCR 抖动）
runs = []
for t, txt in seq:
    if runs and SequenceMatcher(None, runs[-1]["text"], txt).ratio() > 0.86:
        runs[-1]["end"] = t
        if len(txt) > len(runs[-1]["text"]):
            runs[-1]["text"] = txt
    else:
        runs.append({"start": t, "end": t, "text": txt})
runs = [r for r in runs if r["end"] - r["start"] >= 0.4]

print("=== 逐帧 OCR 得到的字幕时间线（标准答案） ===")
for r in runs:
    print(f"  {r['start']:6.2f}-{r['end']:6.2f} ({r['end'] - r['start']:5.2f}s) {r['text']}")

raw = [(t, g) for t, g in stream_band_gray(info, band, fps=6.0) if t <= END]
print(f"\n=== segment() 对比（采样 {len(raw)} 帧，真值 {len(runs)} 条） ===")
for ink in (34, 60):
    for sim in (0.62, 0.72, 0.85):
        cfg = SegmentConfig(ink_thresh=ink, sim_thresh=sim)
        cues, st = segment(list(raw), cfg, fps=6.0, total_duration=END)
        print(f"\n-- ink={ink} sim={sim}: {len(cues)} 条")
        for c in cues:
            print(f"   {c.start:6.2f}-{c.end:6.2f} ({c.duration:5.2f}s)")

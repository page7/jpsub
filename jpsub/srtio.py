"""字幕读取（SRT）与 ASS 输出。

统一用秒（float）在内部表示时间，输出时转成 ASS 时间码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TS = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class Cue:
    """一条字幕。start/end 为秒；source 为原文（日文），text 为译文（中文）。"""

    start: float
    end: float
    source: str = ""
    text: str = ""
    color: str = ""  # 源字幕颜色 #RRGGBB，空表示不保留
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def body(self, mode: str = "target") -> str:
        """按输出模式拼正文。"""
        if mode == "source":
            return self.source or self.text
        if mode == "bilingual":
            if self.source and self.text:
                return f"{self.text}\n{self.source}"
            return self.text or self.source
        return self.text or self.source


def sec_to_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    frac = int(ms) / (10 ** len(ms))
    return int(h) * 3600 + int(m) * 60 + int(s) + frac


def parse_srt(text: str) -> list[Cue]:
    """宽松解析：序号可缺失，时间码支持逗号或点，兼容 CRLF。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        idx = 0
        if re.fullmatch(r"\d+", lines[0].strip()):
            idx = 1
        if idx >= len(lines):
            continue
        m = _TS.search(lines[idx])
        if not m:
            continue
        start = ts_to_sec(*m.group(1, 2, 3, 4))
        end = ts_to_sec(*m.group(5, 6, 7, 8))
        body = "\n".join(lines[idx + 1:]).strip()
        if not body:
            continue
        cues.append(Cue(start=start, end=end, source=body))
    return cues


def read_srt(path: str | Path) -> list[Cue]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "gb18030"):
        try:
            return parse_srt(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return parse_srt(raw.decode("utf-8", errors="replace"))


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,40,&H00FFFFFF,&H000000FF,&H00FFFFFF,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,20,20,28,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_ts(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(hexv: str) -> str:
    hexv = hexv.lstrip("#")
    r, g, b = hexv[0:2], hexv[2:4], hexv[4:6]
    return f"&H{b}{g}{r}&"


def write_ass(cues: list[Cue], path: str | Path, mode: str = "bilingual") -> Path:
    """带颜色的 ASS。颜色按 cue 存的原字幕色；bilingual 时日文行不染色。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_ASS_HEADER]
    for c in cues:
        body = c.body(mode).strip()
        if not body:
            continue
        if mode == "bilingual" and c.source and c.text:
            parts = [c.text, c.source]
            colored = [True, False]
        elif mode == "source":
            parts, colored = [c.source or c.text], [True]
        else:
            parts, colored = [c.text or c.source], [True]
        rendered = []
        for part, do_color in zip(parts, colored):
            if do_color and c.color:
                # 黑字要配白描边，否则直播画面里会糊成一片
                outline = "\\3c&HFFFFFF&" if c.color.upper() == "#000000" else ""
                part = f"{{\\c{_ass_color(c.color)}{outline}}}" + part
            rendered.append(part.replace("\n", "\\N"))
        text = "\\N".join(rendered)
        lines.append(f"Dialogue: 0,{_ass_ts(c.start)},{_ass_ts(c.end)},Default,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


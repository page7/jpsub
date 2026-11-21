"""OCR 后端：把字幕带图片变成日文文本。

paddle  PaddleOCR（默认）。PP-OCRv5 日文模型，纯识别最准，本地跑、不联网
vlm     B 站 Index-Translate 的免费视觉接口。零安装，但会顺手改写原文，只作兜底

两个后端都实现 recognize(img_bgr) -> list[str]，按阅读顺序每行一条。
"""

from __future__ import annotations

import base64
import os
import re
from typing import Callable

import numpy as np

# PaddleOCR 3.x 默认从 HuggingFace 拉权重，国内基本连不上。提前指向国内源。
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_JP_CHAR = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f]")


class OcrError(RuntimeError):
    pass


def looks_japanese(s: str) -> bool:
    return bool(_JP_CHAR.search(s))


def tidy(lines: list[str]) -> str:
    """清掉 OCR 常见噪声：多余空格、零宽字符、纯符号行。"""
    out: list[str] = []
    for ln in lines:
        ln = ln.replace("\u200b", "").replace("\xa0", " ")
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln:
            continue
        if not looks_japanese(ln) and len(ln) < 2:
            continue
        if not looks_japanese(ln) and re.fullmatch(r"[\W_]+", ln):
            continue
        out.append(ln)
    return "\n".join(out)


def _snap_color(r: float, g: float, b: float) -> str:
    """把测得的颜色规整一下：接近白/黑的归到纯白/纯黑，彩色保持原样。"""
    mx, mn = max(r, g, b), min(r, g, b)
    sat = 0.0 if mx <= 0 else (mx - mn) / mx
    if sat < 0.20:  # 灰阶：白字/黑字
        return "#FFFFFF" if mx >= 150 else "#000000"
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _line_color(crop: np.ndarray) -> tuple[str, int]:
    """在一行字幕的裁剪图里找"填充色"和它的像素数。

    字幕通常是「填充 + 描边」两层：描边贴着背景，填充被描边包着。
    所以先切出白/黑/彩色三类像素，再挑面积够大、且离背景最远的那层。
    """
    import cv2

    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    resp = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    mask = resp > 40
    total = int(mask.sum())
    if total < 100:
        return "", 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(int)
    s = hsv[:, :, 1].astype(int)
    bg = (~mask).astype(np.uint8)
    ring = (cv2.dilate(bg, np.ones((3, 3), np.uint8)) > 0) & mask  # 紧贴背景的一圈

    white = mask & (v >= 190) & (s <= 50)
    black = mask & (v <= 80)
    chroma = mask & (s >= 70) & (v >= 90)

    def frac(m: np.ndarray) -> float:
        return float((m & ring).sum()) / max(1, int(m.sum()))

    n_chroma = int(chroma.sum())
    if n_chroma >= total * 0.10:
        # 真填充色一定被描边包着；背景透过来的颜色不挨着字。
        # 用"白/黑强层"做邻接判断，不能用过渡像素，否则会把边缘混色算进来。
        others = (white | black).astype(np.uint8)
        near_others = cv2.dilate(others, np.ones((5, 5), np.uint8)) > 0
        if float((chroma & near_others).sum()) / n_chroma >= 0.55:
            sat = s[chroma]
            strong = chroma & (s >= np.percentile(sat, 60))
            pix = crop[strong] if strong.sum() >= 30 else crop[chroma]
            b, gg, r = (float(np.median(pix[:, i])) for i in range(3))
            return _snap_color(r, gg, b), n_chroma

    if int(black.sum()) >= total * 0.10 and frac(black) < frac(white):
        return "#000000", int(black.sum())
    if int(white.sum()) >= total * 0.10:
        return "#FFFFFF", int(white.sum())
    # 兜底：取面积最大的那层，别让这一条没颜色
    if int(black.sum()) > int(white.sum()):
        return "#000000", int(black.sum())
    if int(white.sum()) > 0:
        return "#FFFFFF", int(white.sum())
    if n_chroma > 0:
        pix = crop[chroma]
        b, gg, r = (float(np.median(pix[:, i])) for i in range(3))
        return _snap_color(r, gg, b), n_chroma
    return "", 0


def detect_text_color(img_bgr: np.ndarray, polys: list | None = None) -> str:
    """识别字幕**填充色**（不是描边），返回 #RRGGBB；判不出来返回空串。

    有 OCR 文字框时逐行分析，取像素最多的那行；没有框时整张图当一行。
    """
    import cv2

    h, w = img_bgr.shape[:2]
    regions: list[np.ndarray] = []
    if polys:
        for poly in polys:
            p = np.asarray(poly)
            x0, y0 = max(0, int(p[:, 0].min())), max(0, int(p[:, 1].min()))
            x1, y1 = min(w, int(p[:, 0].max())), min(h, int(p[:, 1].max()))
            if x1 > x0 + 4 and y1 > y0 + 4:
                regions.append(img_bgr[y0:y1, x0:x1])
    if not regions:
        regions = [img_bgr]

    best, best_n = "", 0
    for crop in regions:
        color, n = _line_color(crop)
        if color and n > best_n:
            best, best_n = color, n
    return best


def _order_lines(items: list[tuple[float, float, str]]) -> list[str]:
    """items = [(cy, cx, text)]，按行聚类后从左到右拼。"""
    if not items:
        return []
    items.sort(key=lambda x: x[0])
    heights = [abs(items[i + 1][0] - items[i][0]) for i in range(len(items) - 1)]
    tol = max(8.0, float(np.median(heights)) * 0.6) if heights else 20.0
    rows: list[list[tuple[float, float, str]]] = []
    for cy, cx, txt in items:
        if rows and abs(cy - rows[-1][0][0]) <= tol:
            rows[-1].append((cy, cx, txt))
        else:
            rows.append([(cy, cx, txt)])
    lines = []
    for row in rows:
        row.sort(key=lambda x: x[1])
        lines.append("".join(t for _, _, t in row).strip())
    return [ln for ln in lines if ln]


class PaddleBackend:
    name = "paddle"

    def __init__(self, lang: str = "japan", log: Callable[[str], None] = print):
        self.log = log
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except Exception as e:
            raise OcrError(
                "PaddleOCR 未安装。先运行 setup.cmd，或改用 --ocr vlm。\n"
                f"原始错误: {e}"
            ) from e

        from paddleocr import PaddleOCR

        self.log("  加载 PaddleOCR 日文模型（首次会下载权重，约 20-80MB）…")
        # 关键：enable_mkldnn=False 必须加。
        # PaddlePaddle 3.x 的 oneDNN 后端在 PIR 执行器下有个崩点，报
        #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
        #        [pir::ArrayAttribute<pir::DoubleAttribute>]  (onednn_instruction.cc)
        # 关掉 mkldnn 就正常，CPU 速度损失很小（字幕带图本来就小）。
        # 不同版本参数名不一致，从"最全"往"最简"逐级降级。
        base = dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False)
        attempts = [
            dict(use_textline_orientation=False, enable_mkldnn=False, **base),
            dict(use_textline_orientation=False, **base),
            base,
        ]
        last: Exception | None = None
        self.ocr = None
        for kw in attempts:
            try:
                self.ocr = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as e:
                last = e
        if self.ocr is None:
            raise OcrError(f"PaddleOCR 初始化失败: {last}")
        self._is_v3 = hasattr(self.ocr, "predict")

    def recognize(self, img_bgr: np.ndarray) -> str:
        self.last_polys: list = []
        if self._is_v3:
            res = self.ocr.predict(img_bgr)
            lines: list[str] = []
            for page in res:
                texts = page.get("rec_texts") or []
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                scores = page.get("rec_scores") or []
                items: list[tuple[float, float, str]] = []
                for i, txt in enumerate(texts):
                    if not txt:
                        continue
                    if scores and i < len(scores) and float(scores[i]) < 0.35:
                        continue
                    poly = np.asarray(polys[i]) if i < len(polys) else None
                    if poly is None or poly.size < 4:
                        items.append((len(items) * 40.0, 0.0, txt))
                        continue
                    self.last_polys.append(poly)
                    xs = poly[:, 0]
                    ys = poly[:, 1]
                    items.append((float(ys.mean()), float(xs.mean()), txt))
                lines.extend(_order_lines(items))
            return tidy(lines)

        # 2.x 路径
        res = self.ocr.ocr(img_bgr, cls=False)
        lines = []
        for page in res or []:
            for det in page or []:
                box, (txt, score) = det[0], det[1]
                if score < 0.35 or not txt:
                    continue
                self.last_polys.append(np.asarray(box, dtype=np.float32))
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                lines.append((sum(ys) / 4, sum(xs) / 4, txt))
        return tidy(_order_lines(lines))


class VlmBackend:
    """B 站免费视觉接口。会改写原文，只适合"看懂大意"的场合。"""

    name = "vlm"

    PROMPT = (
        "逐行输出图片中的日文字幕原文，不要翻译，不要解释，不要加任何标点以外的符号。"
        "一行一条，按从上到下的顺序。如果图中没有日文，输出空。"
    )

    def __init__(self, model: str = "Index-Translate-35B-A3B",
                 base_url: str = "https://index-translate.bilibili.com/v1",
                 api_key: str = "", timeout: float = 120.0,
                 log: Callable[[str], None] = print):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.log = log
        log("  使用 VLM 视觉后端（免费公开端点，可能改写原文）")

    def recognize(self, img_bgr: np.ndarray) -> str:
        import cv2
        import requests

        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
            "Origin": "https://index-translate.bilibili.com",
            "Referer": "https://index-translate.bilibili.com/",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": self.PROMPT},
            ]}],
            "temperature": 0,
        }
        r = requests.post(self.base_url + "/chat/completions",
                          json=payload, headers=headers, timeout=self.timeout)
        if r.status_code != 200:
            raise OcrError(f"VLM OCR HTTP {r.status_code}: {r.text[:200]}")
        content = r.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", content.strip())
        return tidy(content.split("\n"))


def make_backend(kind: str = "paddle", **kw) -> object:
    if kind == "vlm":
        return VlmBackend(**kw)
    if kind == "paddle":
        kw.pop("model", None)
        kw.pop("base_url", None)
        kw.pop("api_key", None)
        kw.pop("timeout", None)
        return PaddleBackend(**kw)
    raise OcrError(f"未知 OCR 后端: {kind}")

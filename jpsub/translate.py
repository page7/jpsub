"""翻译后端：把日文文本批量送出去，拿回中文。

三个后端：
  bilibili  B 站开源的 Index-Translate 公开端点，免费、无需 key（非商用）
  openai    任意 OpenAI 兼容服务（本地 vLLM / 硅基流动 / DeepSeek / 官方 …）
  none      不翻译，只留日文
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import requests

BILI_BASE = "https://index-translate.bilibili.com/v1"

_BATCH_PROMPT = """你是专业的日文字幕译者。把下列日文字条逐条翻成简体中文。

硬性规则：
1. 输出行数必须与输入完全一致，一行对一行，不许合并、拆分、跳过
2. 每行格式固定为：<编号>|||<译文>
3. 不解释、不加括号注释、不写译注、不保留日文原文
4. 保持原句语气强度，口语就译成口语，不要统一成书面语
5. 省略的主语按上下文补全；补不出来就保持无主语（中文允许）
6. 人名、称呼前后一致

输入：
{payload}"""

_SINGLE_PROMPT = """把下面这句日文翻成简体中文，只输出译文，不要任何解释：

{text}"""


class TranslateError(RuntimeError):
    pass


@dataclass
class BackendConfig:
    kind: str = "bilibili"
    model: str = "Index-Translate-35B-A3B"
    base_url: str = BILI_BASE
    api_key: str = ""
    timeout: float = 120.0
    temperature: float = 0.0
    retries: int = 3
    batch_size: int = 16
    concurrency: int = 2
    sleep: float = 0.0


def _post(cfg: BackendConfig, payload: dict) -> str:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # B 站 WAF 会挡掉没有 UA 的请求（实测返回 412）
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    }
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if "bilibili" in cfg.base_url:
        headers["Origin"] = "https://index-translate.bilibili.com"
        headers["Referer"] = "https://index-translate.bilibili.com/"
    last_err = ""
    for attempt in range(cfg.retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except Exception as e:  # 网络抖动、超时
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.2 * (attempt + 1))
    raise TranslateError(last_err or "未知错误")


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
    return s


def _parse_batch(raw: str, n: int) -> dict[int, str]:
    """把 '1|||译文' 逐行解析成 {编号: 译文}。"""
    out: dict[int, str] = {}
    for line in _clean(raw).split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*\|\|\|\s*(.*)$", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
            continue
        m = re.match(r"^(\d+)\s*[.、:：]\s*(.+)$", line)  # 模型偶尔改用点号
        if m and int(m.group(1)) <= n:
            out.setdefault(int(m.group(1)), m.group(2).strip())
    return out


class Translator:
    def __init__(self, cfg: BackendConfig, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.log = log

    # -- 单句
    def one(self, text: str) -> str:
        prompt = _SINGLE_PROMPT.format(text=text)
        raw = _post(self.cfg, {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
        })
        return _clean(raw)

    # -- 一批
    def batch(self, texts: list[str]) -> list[str]:
        payload = "\n".join(f"{i + 1}|||{t}" for i, t in enumerate(texts))
        prompt = _BATCH_PROMPT.format(payload=payload)
        raw = _post(self.cfg, {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
        })
        got = _parse_batch(raw, len(texts))
        # 缺行就地重发单句，不整批重来
        miss = [i for i in range(1, len(texts) + 1) if not got.get(i)]
        if miss:
            self.log(f"    · 批内有 {len(miss)} 条缺失，逐条补发")
            for i in miss:
                try:
                    got[i] = self.one(texts[i - 1])
                except Exception as e:
                    self.log(f"    · 第 {i} 条补发失败: {e}")
        return [got.get(i, "") for i in range(1, len(texts) + 1)]

    # -- 全部
    def run(self, texts: list[str], progress: Callable[[int, int], None] | None = None) -> list[str]:
        n = len(texts)
        if n == 0:
            return []
        bs = max(1, self.cfg.batch_size)
        chunks = [(i, texts[i:i + bs]) for i in range(0, n, bs)]
        results: list[str] = [""] * n
        done = 0

        def work(item: tuple[int, list[str]]) -> None:
            nonlocal done
            off, chunk = item
            try:
                outs = self.batch(chunk)
            except Exception as e:
                self.log(f"  ! 批处理失败，退回逐条: {e}")
                outs = []
                for t in chunk:
                    try:
                        outs.append(self.one(t))
                    except Exception as e2:
                        self.log(f"  ! 单条失败: {e2}")
                        outs.append("")
            for k, v in enumerate(outs):
                results[off + k] = v
            done += len(chunk)
            if progress:
                progress(done, n)
            if self.cfg.sleep:
                time.sleep(self.cfg.sleep)

        workers = max(1, min(self.cfg.concurrency, len(chunks)))
        if workers == 1:
            for it in chunks:
                work(it)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(work, chunks))
        return results


def make_translator(kind: str, model: str | None = None, api_key: str | None = None,
                    base_url: str | None = None, **kw) -> Translator:
    # log 归 Translator 用，不能塞给 BackendConfig
    log = kw.pop("log", print)
    if kind == "none":
        raise TranslateError("none 后端不负责翻译")
    if kind == "openai":
        base = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise TranslateError("openai 后端需要 api_key（或用环境变量 OPENAI_API_KEY）")
        model = model or "gpt-4o-mini"
        cfg = BackendConfig(kind="openai", model=model, base_url=base, api_key=key, **kw)
    else:  # bilibili
        cfg = BackendConfig(
            kind="bilibili",
            model=model or "Index-Translate-35B-A3B",
            base_url=base_url or BILI_BASE,
            api_key=api_key or "",
            **kw,
        )
    return Translator(cfg, log=log)

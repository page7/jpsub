"""字幕变化检测：把连续帧切成一条条 cue。

思路不是"比较整块画面"，而是先把字幕从背景里剥出来再比：

  Top-Hat（提取比背景亮的细结构 = 白字）
  + Black-Hat（提取比背景暗的细结构 = 黑字）
  → 阈值 → 二值图 → 求 IoU

这样画面明暗、镜头移动对判定的影响被压到很低，只有字形真正变化时才判切换。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .srtio import Cue


@dataclass
class SegmentConfig:
    """变化检测参数。一般不用改，遇到灵敏度问题再调。"""

    # 二值化：响应大于该值才算笔画。字幕带偏暗/偏灰时可降到 22 左右
    ink_thresh: int = 34
    # 字幕颜色：auto 不按颜色过滤；white/yellow/black 只保留该颜色的笔画
    # 画面亮、背景纹理多时指定颜色，能明显减少误切
    sub_color: str = "auto"
    # Top-Hat 结构元尺寸（宽, 高）。大于笔画宽度的 2 倍即可
    kernel: tuple[int, int] = (25, 9)
    # 相似度阈值：低于它判为"字幕变了"。0.60~0.75 之间比较稳
    sim_thresh: float = 0.62
    # 需要连续几帧判为变化才真正切（抗单帧噪声/压缩伪影）
    # 4 帧 @6fps ≈ 0.67s：能压住字幕抖动，又不会吞掉正常 2~3 秒的换句
    confirm_frames: int = 4
    # 墨水占比低于此值视为"没有字幕"
    min_ink: float = 0.004
    # 过短的 cue 丢弃或并入邻句
    min_duration: float = 0.24
    # 同一句里允许的最大静止间隔（秒），超过就断开
    max_gap: float = 0.30
    # 参考图缓慢跟随自身（抗轻微抖动/渐显）
    ref_ema: float = 0.75


def _color_keep(bgr: np.ndarray, color: str) -> np.ndarray | None:
    """按字幕颜色返回保留掩码。None 表示不过滤。

    字幕通常是高饱和的纯色块（白/黄/黑），背景多是灰调或杂色，
    按 HSV 思路直接卡 RGB 范围比通用阈值稳。
    """
    if color == "auto":
        return None
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    if color == "white":
        return (mx >= 170) & ((mx - mn) <= 70)
    if color == "yellow":
        return (r >= 150) & (g >= 120) & ((r - b) >= 60)
    if color == "black":
        return mx <= 95
    raise ValueError(f"未知字幕颜色: {color}（可选 auto/white/yellow/black）")


def ink_map(img: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """把字幕带图像转成笔画二值图（0/1 uint8）。

    img 可以是灰度 (HxW) 或 BGR (HxWx3)。传彩图且 cfg.sub_color 不是 auto 时，
    会额外按颜色过滤，只留目标颜色的笔画。
    """
    import cv2

    keep = None
    if img.ndim == 3:
        keep = _color_keep(img, cfg.sub_color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    k = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.kernel)
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    resp = cv2.max(top, black)
    _, bw = cv2.threshold(resp, cfg.ink_thresh, 1, cv2.THRESH_BINARY)
    if keep is not None:
        bw = (bw.astype(bool) & keep).astype(np.uint8)
    # 水平闭运算：把同一行里相邻的笔画连起来，避免笔画抖动被误判为换字
    bw = cv2.morphologyEx(
        bw.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
    )
    return bw



def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两幅笔画图的相似度（IoU，附面积比门限）。"""
    sa = int(a.sum())
    sb = int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    area_ratio = min(sa, sb) / max(sa, sb)
    inter = int(np.logical_and(a, b).sum())
    union = sa + sb - inter
    iou = inter / union if union else 1.0
    # 一行字 vs 两行字，IoU 可能还不低，但面积差明显 —— 强制判不同
    if area_ratio < 0.40:
        return min(iou, 0.30)
    return iou


@dataclass
class SegmentStats:
    sampled: int = 0
    cues: int = 0
    blank_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)


def segment(
    frames: list[tuple[float, np.ndarray]] | "object",
    cfg: SegmentConfig | None = None,
    fps: float = 6.0,
    total_duration: float | None = None,
) -> tuple[list[Cue], SegmentStats]:
    """输入 (时间, 灰度图) 序列，输出带起止时间的空 cue 列表。"""
    cfg = cfg or SegmentConfig()
    stats = SegmentStats()
    cues: list[Cue] = []

    ref: np.ndarray | None = None
    cur_start: float | None = None
    cur_end: float | None = None
    last_hit: float | None = None
    pending = 0
    pending_start: float | None = None
    blank = 0

    for t, gray in frames:
        stats.sampled += 1
        bw = ink_map(gray, cfg)
        ink = float(bw.mean())

        if ink < cfg.min_ink:
            blank += 1
            # 字幕消失 → 结束当前 cue
            if cur_start is not None and last_hit is not None:
                cues.append(Cue(start=cur_start, end=max(last_hit, cur_start + 0.04)))
                cur_start = ref = None
                pending = 0
            continue

        if cur_start is None:
            cur_start = t
            cur_end = t
            last_hit = t
            ref = bw
            continue

        assert ref is not None
        sim = similarity(bw, ref)

        if sim >= cfg.sim_thresh:
            pending = 0
            pending_start = None
            last_hit = t
            cur_end = t
            # 参考图缓慢跟随，吸收渐显/微抖动
            if cfg.ref_ema > 0:
                ref = (cfg.ref_ema * ref + (1 - cfg.ref_ema) * bw).astype(np.uint8)
                ref = (ref > 0.5).astype(np.uint8)
            continue

        # 相似度掉下来：先攒确认帧，避免被压缩噪声骗
        if pending == 0:
            pending_start = last_hit if last_hit is not None else t
        pending += 1
        if pending < cfg.confirm_frames:
            continue

        # 确认切换。边界取"最后一次相同"的下一帧，减半帧做补偿
        assert pending_start is not None
        boundary = min(t, pending_start + 1.0 / max(fps, 0.001) * 0.5)
        if last_hit is not None:
            cues.append(Cue(start=cur_start, end=max(last_hit, cur_start + 0.04)))
        cur_start = boundary
        cur_end = t
        last_hit = t
        ref = bw
        pending = 0
        pending_start = None

    if cur_start is not None and last_hit is not None:
        cues.append(Cue(start=cur_start, end=max(last_hit, cur_start + 0.04)))

    stats.blank_ratio = blank / stats.sampled if stats.sampled else 1.0
    cues = _cleanup(cues, cfg, total_duration)
    stats.cues = len(cues)
    return cues, stats


def _cleanup(cues: list[Cue], cfg: SegmentConfig, total_duration: float | None) -> list[Cue]:
    """丢掉过短/空白的 cue，并把紧邻的同类碎片合起来。"""
    if not cues:
        return []

    out: list[Cue] = []
    for c in cues:
        if total_duration is not None:
            c.end = min(c.end, total_duration)
        if c.duration < cfg.min_duration:
            # 太短：多数是淡入淡出或噪声。若紧挨着前一条就并进去，否则丢弃
            if out and c.start - out[-1].end < cfg.max_gap:
                out[-1].end = max(out[-1].end, c.end)
                continue
            if not out:
                continue
        out.append(c)

    # 时间轴不能重叠或倒序
    for i in range(1, len(out)):
        if out[i].start < out[i - 1].end:
            out[i].start = out[i - 1].end
        if out[i].end <= out[i].start:
            out[i].end = out[i].start + 0.05

    return [c for c in out if c.end > c.start]

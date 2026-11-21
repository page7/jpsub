"""视频探测与抽帧。

两条链路：
  - stream_band_gray : 一次性解码全片，只取「字幕带」的灰度缩略图，用于变化检测（快）
  - grab_band_color  : 精确 seek 取某一时刻的字幕带彩色图，用于 OCR（慢但准）

优先用 ffmpeg（imageio-ffmpeg 自带一份，无需系统安装），失败则退回 OpenCV。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

_FFMPEG_CACHE: str | None | bool = False
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _hidden_kwargs() -> dict:
    """让 ffmpeg 子进程不弹控制台窗口（Windows 下 GUI 调用尤其重要）。"""
    if os.name != "nt":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": _NO_WINDOW, "startupinfo": si}


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """写图片，兼容 Windows 中文/Unicode 路径。

    cv2.imwrite 在中文路径下可能静默返回 False，改用 imencode + Path.write_bytes。
    """
    import cv2

    p = Path(path)
    ext = p.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        ext = ".png"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        p.write_bytes(buf.tobytes())
        return True
    except OSError:
        return False



def find_ffmpeg() -> str | None:
    """系统 PATH 里的 ffmpeg 优先，其次 imageio-ffmpeg 自带的。找不到返回 None。"""
    global _FFMPEG_CACHE
    if _FFMPEG_CACHE is not False:
        return _FFMPEG_CACHE  # type: ignore[return-value]
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and not Path(exe).exists():
                exe = None
        except Exception:
            exe = None
    _FFMPEG_CACHE = exe
    return exe


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float

    def __str__(self) -> str:
        return (
            f"{self.path.name}  {self.width}x{self.height}  "
            f"{self.fps:.3f}fps  {self.duration:.1f}s ({self.n_frames} 帧)"
        )


def probe(path: str | Path) -> VideoInfo:
    import cv2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"视频不存在: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or fps > 1000:
        fps = 25.0
    duration = n / fps if n > 0 else 0.0
    if duration <= 0:
        raise RuntimeError(f"无法读取时长，视频可能损坏: {path}")
    if w <= 0 or h <= 0:
        raise RuntimeError(f"无法读取分辨率: {path}")
    return VideoInfo(path=path, width=w, height=h, fps=fps, n_frames=n, duration=duration)


# ---------------------------------------------------------------- 几何计算


def band_geometry(
    width: int, height: int, band: tuple[float, float], target_w: int = 960
) -> tuple[int, int, int, int]:
    """把相对字幕带 (y0, y1) 换算成 ffmpeg crop 参数 (cw, ch, cx, cy) 与输出尺寸。

    返回 (cw, ch, cx, cy)；输出尺寸由 caller 用 scale 参数算。
    """
    y0, y1 = band
    y0 = max(0.0, min(1.0, y0))
    y1 = max(0.0, min(1.0, y1))
    if y1 <= y0:
        raise ValueError(f"字幕带区间非法: {band}")
    cw = width
    ch = int(height * (y1 - y0))
    cy = int(height * y0)
    # yuv420 要求偶数，统一抹平，避免 ffmpeg 报 "Invalid crop" 或输出偏一像素
    ch = max(2, ch // 2 * 2)
    cy = min(cy // 2 * 2, height - ch)
    return cw, ch, 0, max(0, cy)


def scaled_size(cw: int, ch: int, target_w: int) -> tuple[int, int]:
    ow = max(2, min(target_w, cw) // 2 * 2)
    oh = max(2, int(round(ch * ow / cw)) // 2 * 2)
    return ow, oh


# ---------------------------------------------------------------- 抽帧


def _stream_band(
    info: VideoInfo,
    band: tuple[float, float],
    fps: float,
    target_w: int,
    pix_fmt: str,
    channels: int,
) -> Iterator[tuple[float, np.ndarray]]:
    """按 fps 采样，逐帧产出 (时间秒, 字幕带图像)。

    ffmpeg 不可用时退回 cv2 顺序解码（内存占用低，但慢一些）。
    """
    cw, ch, cx, cy = band_geometry(info.width, info.height, band, target_w)
    ow, oh = scaled_size(cw, ch, target_w)
    exe = find_ffmpeg()
    shape = (oh, ow) if channels == 1 else (oh, ow, channels)

    if exe:
        vf = f"fps={fps},crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh}"
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(info.path),
            "-an", "-sn", "-dn",
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 **_hidden_kwargs())
        size = ow * oh * channels
        i = 0
        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(size)
                if not buf or len(buf) < size:
                    break
                yield i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(shape)
                i += 1
        finally:
            proc.stdout and proc.stdout.close()
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            proc.stderr and proc.stderr.close()
            proc.wait()
            if i == 0 and err.strip():
                raise RuntimeError(f"ffmpeg 抽帧失败:\n{err.strip()[:800]}")
        return

    # ---- OpenCV 兜底
    import cv2

    cap = cv2.VideoCapture(str(info.path))
    step = max(1, int(round(info.fps / fps)))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                crop = frame[cy:cy + ch, cx:cx + cw]
                if channels == 1:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                yield idx / info.fps, cv2.resize(crop, (ow, oh),
                                                interpolation=cv2.INTER_AREA)
            idx += 1
    finally:
        cap.release()


def stream_band_gray(
    info: VideoInfo,
    band: tuple[float, float],
    fps: float = 6.0,
    target_w: int = 960,
) -> Iterator[tuple[float, np.ndarray]]:
    """按 fps 采样，逐帧产出 (时间秒, 字幕带灰度图 HxW uint8)。"""
    return _stream_band(info, band, fps, target_w, "gray", 1)


def stream_band_color(
    info: VideoInfo,
    band: tuple[float, float],
    fps: float = 6.0,
    target_w: int = 960,
) -> Iterator[tuple[float, np.ndarray]]:
    """同 stream_band_gray，但输出 BGR 三通道（按字幕颜色过滤时用）。"""
    return _stream_band(info, band, fps, target_w, "bgr24", 3)



def grab_band_color(
    info: VideoInfo,
    t: float,
    band: tuple[float, float],
    target_w: int = 1280,
) -> np.ndarray | None:
    """取 t 秒处字幕带的彩色图（BGR）。ffmpeg 精确 seek，失败退回 cv2。"""
    cw, ch, cx, cy = band_geometry(info.width, info.height, band, target_w)
    ow, oh = scaled_size(cw, ch, target_w)
    exe = find_ffmpeg()

    if exe:
        vf = f"crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh}"
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{max(0.0, t):.3f}", "-i", str(info.path),
            "-frames:v", "1", "-an", "-sn", "-dn",
            "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=60,
                                 **_hidden_kwargs())
        except subprocess.TimeoutExpired:
            return None
        if out.returncode == 0 and out.stdout:
            import cv2

            buf = np.frombuffer(out.stdout, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        return None

    import cv2

    cap = cv2.VideoCapture(str(info.path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return None
        crop = frame[cy:cy + ch, cx:cx + cw]
        return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA)
    finally:
        cap.release()

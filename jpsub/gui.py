"""jpsub 的简单图形界面。

不重复实现流程，只负责拼命令行、启动子进程、把 stdout 里的
`[阶段]` / `[###] n/N` 解析成进度条。核心代码仍然在 cli/pipeline 里。
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from codecs import getincrementaldecoder
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path.home() / ".jpsub_gui.json"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_STAGE = re.compile(r"\[(\d)/4\]\s*(.*)")
_PROGRESS = re.compile(r"\[[#.]+\]\s*(\d+)/(\d+)(?:.*?剩余约\s*([\d.]+)s)?")
_DONE = re.compile(r"完成，共\s*(\d+)\s*条")
_NOISE = (
    "INFO: Could not find files",
    "Creating model:",
    "Model files already exist",
    "No ccache found",
    "warnings.warn",
    "extension_utils.py",
)

_VIDEO_TYPES = [
    ("视频文件", "*.mp4 *.mkv *.ts *.m2ts *.avi *.mov *.webm *.flv *.wmv"),
    ("所有文件", "*.*"),
]


def _hidden_kwargs() -> dict:
    if os.name != "nt":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0), "startupinfo": si}


def _fmt_eta(sec: str | None) -> str:
    if not sec:
        return ""
    try:
        n = int(float(sec))
    except ValueError:
        return ""
    if n >= 60:
        return f"剩余约 {n // 60} 分 {n % 60} 秒"
    return f"剩余约 {n} 秒"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("jpsub — 视频字幕生成")
        self.geometry("1120x780")
        self.minsize(920, 700)

        self.q: queue.Queue[tuple[str, object]] = queue.Queue()
        self.proc: subprocess.Popen[bytes] | None = None
        self.reader: threading.Thread | None = None
        self.running = False
        self.saw_done = False
        self.stopping = False
        self.mode = "run"
        self.locate_dir: Path | None = None
        self.editable: list[tuple[tk.Widget, str]] = []

        self._init_vars()
        self._build()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll)

    # ------------------------------------------------------------ 变量/默认值

    def _init_vars(self) -> None:
        defaults = {
            "video": "",
            "out": "",
            "band": "0.80,0.94",
            "sub_color": "white",
            "ocr": "paddle",
            "fps": "6",
            "sim": "0.62",
            "ink": "50",
            "min_dur": "0.6",
            "confirm": "4",
            "max_line": "20",
            "max_seconds": "",
            "cues_dir": "",
            "translate": "bilibili",
            "model": "",
            "base_url": "",
            "api_key": "",
            "batch": "16",
            "concurrency": "2",
        }
        self.vars: dict[str, tk.StringVar] = {k: tk.StringVar(value=v) for k, v in defaults.items()}

    # ------------------------------------------------------------ 界面

    def _build(self) -> None:
        style = ttk.Style(self)
        for name in ("vista", "winnative", "clam"):
            if name in style.theme_names():
                style.theme_use(name)
                break
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        # 文件
        files = ttk.LabelFrame(root, text="文件", padding=8)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self._label(files, "视频", 0)
        e = ttk.Entry(files, textvariable=self.vars["video"])
        e.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        self._reg(e)
        self._button(files, "选择视频…", self._pick_video).grid(row=0, column=2, pady=3)

        self._label(files, "输出目录（留空=视频目录）", 1)
        e = ttk.Entry(files, textvariable=self.vars["out"])
        e.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        self._reg(e)
        self._button(files, "选择目录…", self._pick_out).grid(row=1, column=2, pady=3)

        # 参数
        params = ttk.Frame(root)
        params.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        params.columnconfigure(0, weight=1)
        params.columnconfigure(1, weight=1)

        left = ttk.LabelFrame(params, text="切分与识别", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.columnconfigure(1, weight=1)
        right = ttk.LabelFrame(params, text="翻译", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.columnconfigure(1, weight=1)

        row = 0
        self._label(left, "字幕带", row)
        e = ttk.Entry(left, textvariable=self.vars["band"], width=16)
        e.grid(row=row, column=1, sticky="w", padx=6, pady=2)
        self._reg(e)
        ttk.Label(left, text="如 0.80,0.94 或 auto", foreground="#777").grid(
            row=row, column=2, sticky="w", padx=(0, 4))
        row += 1
        row = self._combo(left, row, "字幕颜色", "sub_color",
                          ("auto", "white", "yellow", "black"))
        row = self._combo(left, row, "OCR 后端", "ocr", ("paddle", "vlm"))
        row = self._entry(left, row, "采样帧率", "fps", "每秒采样次数，默认 6")
        row = self._entry(left, row, "切换阈值", "sim", "默认 0.62")
        row = self._entry(left, row, "笔画阈值", "ink", "默认 34，白字可 50")
        row = self._entry(left, row, "最短时长", "min_dur", "秒，默认 0.24")
        row = self._entry(left, row, "确认帧数", "confirm", "默认 4，抗字幕抖动")
        row = self._entry(left, row, "单行字数", "max_line", "0 表示不自动断行")
        row = self._entry(left, row, "只跑前 N 秒", "max_seconds", "留空为全片")
        self._label(left, "cue 截图目录", row)
        e = ttk.Entry(left, textvariable=self.vars["cues_dir"])
        e.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        self._reg(e)
        self._button(left, "…", self._pick_cues, width=3).grid(row=row, column=2, pady=2)
        ttk.Label(left, text="留空 = 输出目录/cues", foreground="#777").grid(
            row=row + 1, column=1, sticky="w", padx=6)
        row += 2

        row = 0
        row = self._combo(right, row, "翻译后端", "translate",
                          ("bilibili", "openai", "none"))
        row = self._entry(right, row, "模型", "model", "留空用默认")
        row = self._entry(right, row, "Base URL", "base_url", "openai 兼容地址")
        self._label(right, "API Key", row)
        e = ttk.Entry(right, textvariable=self.vars["api_key"], show="•")
        e.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        self._reg(e)
        row += 1
        row = self._entry(right, row, "每批条数", "batch", "默认 16")
        row = self._entry(right, row, "并发批数", "concurrency", "默认 2")
        ttk.Label(right, text="openai 后端需要 Base URL 和 API Key", foreground="#777").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # 操作按钮
        btns = ttk.Frame(root)
        btns.grid(row=2, column=0, sticky="ew", pady=8)
        self.start_btn = ttk.Button(btns, text="开始生成", style="Accent.TButton",
                                    command=self._start)
        self.start_btn.pack(side="left")
        self.locate_btn = ttk.Button(btns, text="定位测试", command=self._locate)
        self.locate_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(btns, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="打开输出目录", command=self._open_out).pack(side="left", padx=6)
        ttk.Button(btns, text="打开 cue 目录", command=self._open_cues).pack(side="left")
        ttk.Button(btns, text="打开定位目录", command=self._open_locate).pack(side="left", padx=6)
        ttk.Button(btns, text="清空日志", command=self._clear_log).pack(side="right")

        # 进度 + 日志
        bottom = ttk.Frame(root)
        bottom.grid(row=3, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(2, weight=1)

        prog = ttk.Frame(bottom)
        prog.grid(row=0, column=0, sticky="ew")
        prog.columnconfigure(1, weight=1)
        self.stage_var = tk.StringVar(value="等待开始")
        self.pct_var = tk.StringVar(value="")
        self.eta_var = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self.stage_var, font=("Microsoft YaHei UI", 10, "bold")).grid(
            row=0, column=0, sticky="w")
        self.pb = ttk.Progressbar(prog, mode="determinate", maximum=100, value=0)
        self.pb.grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Label(prog, textvariable=self.pct_var, width=18, anchor="e").grid(row=0, column=2, sticky="e")
        ttk.Label(prog, textvariable=self.eta_var, foreground="#666").grid(
            row=1, column=1, sticky="w", padx=10)

        log_frame = ttk.LabelFrame(bottom, text="运行日志", padding=4)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="none", state="disabled",
                            font=("Consolas", 9), background="#FAFAFA",
                            relief="flat", padx=6, pady=4)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)

    def _label(self, parent: tk.Widget, text: str, row: int) -> None:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=2)

    def _entry(self, parent: tk.Widget, row: int, label: str, key: str, hint: str = "") -> int:
        self._label(parent, label, row)
        e = ttk.Entry(parent, textvariable=self.vars[key], width=18)
        e.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        self._reg(e)
        if hint:
            ttk.Label(parent, text=hint, foreground="#777").grid(
                row=row, column=2, sticky="w", padx=(0, 4))
        return row + 1

    def _combo(self, parent: tk.Widget, row: int, label: str, key: str,
               values: tuple[str, ...]) -> int:
        self._label(parent, label, row)
        c = ttk.Combobox(parent, textvariable=self.vars[key], values=values,
                         state="readonly", width=14)
        c.grid(row=row, column=1, sticky="w", padx=6, pady=2)
        self._reg(c, "readonly")
        return row + 1

    def _button(self, parent: tk.Widget, text: str, command, width: int | None = None) -> ttk.Button:
        b = ttk.Button(parent, text=text, command=command)
        if width:
            b.configure(width=width)
        return b

    def _reg(self, widget: tk.Widget, state: str = "normal") -> None:
        self.editable.append((widget, state))

    # ------------------------------------------------------------ 文件选择

    def _pick_video(self) -> None:
        p = filedialog.askopenfilename(title="选择视频", filetypes=_VIDEO_TYPES)
        if p:
            self.vars["video"].set(p)

    def _pick_out(self) -> None:
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.vars["out"].set(p)

    def _pick_cues(self) -> None:
        p = filedialog.askdirectory(title="选择 cue 截图目录")
        if p:
            self.vars["cues_dir"].set(p)

    # ------------------------------------------------------------ 参数校验/命令行

    def _num(self, key: str, label: str, *, integer: bool = False,
             min_value: float | None = None, max_value: float | None = None) -> str | None:
        s = self.vars[key].get().strip()
        if not s:
            return None
        try:
            v = int(s) if integer else float(s)
        except ValueError:
            raise ValueError(f"{label} 不是有效数字：{s}") from None
        if min_value is not None and v < min_value:
            raise ValueError(f"{label} 不能小于 {min_value}")
        if max_value is not None and v > max_value:
            raise ValueError(f"{label} 不能大于 {max_value}")
        return str(v)

    def _video_path(self) -> str:
        video = self.vars["video"].get().strip().strip('"')
        if not video:
            raise ValueError("请先选择视频文件")
        if not Path(video).exists():
            raise ValueError(f"视频不存在：{video}")
        return video

    def _band_value(self) -> str:
        band = self.vars["band"].get().strip()
        if band and band.lower() != "auto":
            parts = band.replace("，", ",").split(",")
            try:
                a, b = (float(x) for x in parts)
            except Exception:
                raise ValueError("字幕带格式应为 0.80,0.94，或填 auto") from None
            if not (0 <= a < b <= 1):
                raise ValueError("字幕带需要满足 0 ≤ 上边界 < 下边界 ≤ 1")
        return band

    def build_args(self) -> list[str]:
        video = self._video_path()

        args = [sys.executable, "-u", "-m", "jpsub", "run", video]

        out = self.vars["out"].get().strip().strip('"')
        if out:
            args += ["-o", out]
        band = self._band_value()
        if band:
            args += ["--band", band]
        args += ["--sub-color", self.vars["sub_color"].get()]
        args += ["--ocr", self.vars["ocr"].get()]

        for key, flag, label, integer, lo, hi in (
            ("fps", "--fps", "采样帧率", False, 0.1, 60),
            ("sim", "--sim", "切换阈值", False, 0.0, 1.0),
            ("ink", "--ink", "笔画阈值", True, 0, 255),
            ("min_dur", "--min-dur", "最短时长", False, 0.0, 60),
            ("confirm", "--confirm", "确认帧数", True, 1, 60),
            ("max_line", "--max-line", "单行字数", True, 0, 1000),
        ):
            v = self._num(key, label, integer=integer, min_value=lo, max_value=hi)
            if v is not None:
                args += [flag, v]

        max_sec = self._num("max_seconds", "只跑前 N 秒", min_value=0.1)
        if max_sec is not None:
            args += ["--max-seconds", max_sec]
        cues = self.vars["cues_dir"].get().strip().strip('"')
        if cues:
            args += ["--debug-dir", cues]

        args += ["--translate", self.vars["translate"].get()]
        for key, flag in (("model", "--model"), ("base_url", "--base-url"),
                          ("api_key", "--api-key")):
            v = self.vars[key].get().strip()
            if v:
                args += [flag, v]
        for key, flag, label, lo in (("batch", "--batch", "每批条数", 1),
                                     ("concurrency", "--concurrency", "并发批数", 1)):
            v = self._num(key, label, integer=True, min_value=lo)
            if v is not None:
                args += [flag, v]

        return args

    def build_locate_args(self) -> list[str]:
        video = self._video_path()
        band = self._band_value()
        out = self._out_dir() or Path(video).parent
        self.locate_dir = out / "locate"
        args = [sys.executable, "-u", "-m", "jpsub", "locate", video,
                "--out", str(self.locate_dir), "--count", "6"]
        if band:
            args += ["--band", band]
        args += ["--ocr", self.vars["ocr"].get()]
        return args

    @staticmethod
    def _display_args(args: list[str]) -> str:
        shown: list[str] = []
        hide = False
        for a in args:
            if hide:
                shown.append("***")
                hide = False
                continue
            shown.append(a)
            if a == "--api-key":
                hide = True
        return subprocess.list2cmdline(shown)

    # ------------------------------------------------------------ 运行

    def _start(self) -> None:
        if self.running:
            return
        try:
            args = self.build_args()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        self._launch(args, "run")

    def _locate(self) -> None:
        if self.running:
            return
        try:
            args = self.build_locate_args()
        except ValueError as e:
            messagebox.showerror("参数有误", str(e))
            return
        self._launch(args, "locate")

    def _launch(self, args: list[str], mode: str) -> None:
        self.mode = mode
        self._save_settings()
        self._clear_progress()
        self.stage_var.set("定位测试…" if mode == "locate" else "启动中…")
        self._log("")
        title = "定位测试" if mode == "locate" else "新的任务"
        self._log(f"===== {time.strftime('%H:%M:%S')} {title} =====")
        self._log("命令：" + self._display_args(args))
        self.saw_done = False
        self.stopping = False

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            self.proc = subprocess.Popen(
                args, cwd=str(ROOT), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, **_hidden_kwargs(),
            )
        except Exception as e:
            self._log(f"启动失败：{e}")
            self.stage_var.set("启动失败")
            messagebox.showerror("启动失败", str(e))
            return

        self._set_running(True)
        self.reader = threading.Thread(target=self._read_output, args=(self.proc,), daemon=True)
        self.reader.start()

    def _read_output(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        dec = getincrementaldecoder("utf-8")("replace")
        buf = ""
        fd = proc.stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf += dec.decode(chunk)
                parts = re.split(r"[\r\n]", buf)
                buf = parts.pop()
                for line in parts:
                    if line.strip():
                        self.q.put(("line", line))
        except Exception as e:
            self.q.put(("line", f"读取输出失败：{e}"))
        finally:
            buf += dec.decode(b"", True)
            if buf.strip():
                self.q.put(("line", buf))
            code = proc.wait()
            self.q.put(("exit", code))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.q.get_nowait()
                if kind == "line":
                    self._handle_line(str(value))
                elif kind == "exit":
                    self._on_exit(int(value))  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _handle_line(self, raw: str) -> None:
        line = _ANSI.sub("", raw).replace("\x00", "").rstrip()
        if not line.strip():
            return
        stripped = line.strip()
        if any(s in stripped for s in _NOISE):
            return

        m = _STAGE.search(stripped)
        if m:
            self.stage_var.set(f"阶段 {m.group(1)}/4 · {m.group(2)}")
            self._clear_progress()
            self._log(stripped)
            return

        m = _PROGRESS.search(stripped)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            self._set_progress(done, total, m.group(3))
            return

        m = _DONE.search(stripped)
        if m:
            self.saw_done = True
            self._finish_progress(f"完成，共 {m.group(1)} 条")
            self._log(stripped)
            return

        if "完成，共" in stripped:
            self.saw_done = True
        self._log(stripped)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        # 日志太长时丢掉前半段，避免 Text 越跑越慢
        try:
            if int(self.log.index("end-1c").split(".")[0]) > 4000:
                self.log.delete("1.0", "1500.0")
        except tk.TclError:
            pass
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _clear_progress(self) -> None:
        self.pb.stop()
        self.pb.configure(mode="indeterminate", maximum=100, value=0)
        self.pb.start(12)
        self.pct_var.set("")
        self.eta_var.set("")

    def _set_progress(self, done: int, total: int, eta: str | None) -> None:
        self.pb.stop()
        self.pb.configure(mode="determinate", maximum=max(1, total), value=done)
        pct = done / total * 100 if total else 0.0
        self.pct_var.set(f"{done}/{total}  {pct:5.1f}%")
        self.eta_var.set(_fmt_eta(eta))

    def _finish_progress(self, text: str) -> None:
        self.pb.stop()
        self.pb.configure(mode="determinate", maximum=100, value=100)
        self.pct_var.set("100%")
        self.eta_var.set("")
        self.stage_var.set(text)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.start_btn.configure(state="disabled" if running else "normal")
        self.locate_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        for widget, normal in self.editable:
            try:
                widget.configure(state="disabled" if running else normal)
            except tk.TclError:
                pass

    def _on_exit(self, code: int) -> None:
        self.proc = None
        mode = self.mode
        self._set_running(False)
        if self.stopping:
            self.stage_var.set("已停止")
            self._log("任务已停止")
            self.stopping = False
            return
        if code == 0:
            if mode == "locate":
                self._finish_progress("定位测试完成")
                self._log("定位测试完成")
                sheet = self.locate_dir / "locate_sheet.png" if self.locate_dir else None
                if sheet and sheet.exists():
                    try:
                        os.startfile(str(sheet))  # type: ignore[attr-defined]
                    except Exception as e:
                        self._log(f"打开定位拼图失败：{e}")
                return
            if not self.saw_done:
                self.stage_var.set("结束")
                self._finish_progress("结束")
            self._log("任务结束")
        else:
            self.pb.stop()
            self.stage_var.set(f"失败（退出码 {code}）")
            self.pct_var.set("")
            self.eta_var.set("")
            self._log(f"任务失败，退出码 {code}")
            messagebox.showerror("任务失败", f"进程退出码 {code}\n请查看运行日志。")

    def _stop(self, confirm: bool = True) -> None:
        if not self.running or self.proc is None:
            return
        if confirm and not messagebox.askyesno("停止任务", "确定要停止当前任务吗？"):
            return
        self.stopping = True
        self.stage_var.set("正在停止…")
        self._log("正在停止进程…")
        pid = self.proc.pid
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, **_hidden_kwargs())
        except Exception:
            try:
                self.proc.kill()
            except OSError:
                pass

    # ------------------------------------------------------------ 打开目录/配置

    def _out_dir(self) -> Path | None:
        out = self.vars["out"].get().strip().strip('"')
        if out:
            return Path(out)
        video = self.vars["video"].get().strip().strip('"')
        if video:
            return Path(video).parent
        return None

    def _cues_dir(self) -> Path | None:
        cues = self.vars["cues_dir"].get().strip().strip('"')
        if cues:
            return Path(cues)
        out = self._out_dir()
        return out / "cues" if out else None

    def _locate_dir(self) -> Path | None:
        out = self._out_dir()
        return out / "locate" if out else self.locate_dir

    def _open_out(self) -> None:
        self._open_dir(self._out_dir())

    def _open_cues(self) -> None:
        self._open_dir(self._cues_dir())

    def _open_locate(self) -> None:
        self._open_dir(self._locate_dir())

    def _open_dir(self, path: Path | None) -> None:
        if path is None or not path.exists():
            messagebox.showinfo("提示", "目录还没生成。")
            return
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def _load_settings(self) -> None:
        if not CONFIG.exists():
            return
        try:
            data = json.loads(CONFIG.read_text("utf-8"))
        except Exception:
            return
        for key, var in self.vars.items():
            if key in data and isinstance(data[key], str):
                var.set(data[key])

    def _save_settings(self) -> None:
        data = {key: var.get() for key, var in self.vars.items() if key != "api_key"}
        try:
            CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
        except OSError:
            pass

    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("退出", "任务还在运行，确定停止并退出吗？"):
                return
            self._stop(confirm=False)
        self._save_settings()
        self.destroy()


def main() -> int:
    try:
        app = App()
        app.mainloop()
    except Exception:
        import tempfile
        import traceback

        detail = traceback.format_exc()
        try:
            log = Path(tempfile.gettempdir()) / "jpsub_gui_error.log"
            log.write_text(detail, encoding="utf-8")
            messagebox.showerror("jpsub 启动失败", f"{detail}\n\n日志：{log}")
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

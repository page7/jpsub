# jpsub — 日文字幕视频 → 中文 ASS

把带日文硬字幕的视频，转成带原字幕颜色的中文 ASS。全本地跑，除了翻译那一步。

```
视频 ──抽帧──> 字幕带 ──变化检测──> cue 列表 ──OCR──> 日文 ──翻译──> 中文 ASS
```

## 安装

已验证：**PaddleOCR 3.7 + Python 3.12**。

```bat
D:\ai\jpsub\setup.cmd
```

`setup.cmd` 会自动找 Python 3.12 建 `.venv`、装依赖（含 PaddleOCR，约 400MB）。

> 为什么指定 3.12：PaddlePaddle 对 Python 3.13 的 wheel 不稳定，3.12 最稳。

## 图形界面

不想敲命令就双击：

```bat
jpsub-gui.cmd
```

界面上可以：

- 选择视频、输出目录、cue 截图目录
- 设置字幕带、字幕颜色、OCR 后端、采样帧率、切换阈值、最短时长、确认帧数
- 设置翻译后端（bilibili / openai / none）、模型、Base URL、API Key、批大小和并发
- 点击「开始生成」后实时查看阶段、进度条、剩余时间、运行日志
- 点击「定位测试」只抓 6 帧，自动挑有字幕的画面并拼成 `locate_sheet.png`，不用等整片
- 「停止」会连同 ffmpeg 子进程一起结束；跑完可直接打开输出目录 / cue 目录 / 定位目录

参数和日志仍和命令行完全一致，GUI 只是拼参数并启动子进程。界面会记住上次设置，但不会保存 API Key。

## 三步用法

### 1. 先看字幕在哪（10 秒）

```bat
jpsub.cmd probe "D:\videos\某番剧.mp4"
```

输出建议字幕带，比如 `--band 0.81,0.97`。**这一步别跳过**——字幕带选错，后面全废。

想眼见为实，先跑定位测试（推荐）：

```bat
jpsub.cmd locate "D:\videos\某番剧.mp4" --band 0.81,0.97
```

它会扫一圈，挑出有字幕的 6 帧，存到视频目录下的 `locate/`，并拼成一张 `locate_sheet.png`。打开拼图：字幕要**完整**在框内，**不能**带上台标、进度条、弹幕。也可以只截一帧：

```bat
jpsub.cmd ocr "D:\videos\某番剧.mp4" -t 60 --band 0.81,0.97 --save band.png
```

`locate` 常用参数：

```bat
--count 6          抓几帧，默认 6
--times 60,300,900 指定秒数，跳过自动挑帧
--ocr none         只存图，不跑 OCR
-o DIR             输出目录，默认视频目录/locate
```

### 2. 先试 2 分钟，别直接整片跑

```bat
jpsub.cmd "D:\videos\某番剧.mp4" --band 0.81,0.97 --max-seconds 120
```

看输出的 `.ass` 准不准。不准的话按下面的表调参。输出目录下的 `cues/` 里是每条 cue 的截图，对不上时直接翻图找原因。

### 3. 整片跑

```bat
jpsub.cmd "D:\videos\某番剧.mp4" --band 0.81,0.97
```

产物（默认输出到视频同目录）：

| 文件 | 内容 |
|---|---|
| `xxx.ass` | 纯中文，自动带原字幕填充色（压片/播放最稳） |
| `cues/` | 每条最终输出 cue 的字幕带截图，用于对比验收（默认保存） |

> 不再输出 SRT / 纯文本 / JSON；需要别的格式请从 `.ass` 转换。

## 字幕带选错怎么调

| 现象 | 原因 | 怎么改 |
|---|---|---|
| 一条字幕都没切出来 | 带子完全没框到字 | 跑 `probe` 重选 |
| 字幕被切成好几条碎片 | 太敏感 | `--sim 0.50` |
| 相邻两句被并成一条 | 不够敏感 | `--sim 0.72` |
| 一句字幕中间断掉 | 淡入淡出被判成切换 | `--min-dur 0.35`、`--fps 8` |
| 切出一堆空 cue | 带子里有台标/水印 | 缩窄 band，避开台标高度 |
| 字幕识别成空白 | 字幕不是白字（黄字/黑字） | 黄字 `--sub-color yellow`；黑字 `--sub-color black` 或 `--ink 22` |
| 背景亮/很花，误切特别多 | 掩码把背景算成笔画 | 指定字幕颜色，如 `--sub-color white` |
| 两行字幕只读出一行 | band 没盖住第二行 | 加高 band，例如 `--band 0.80,0.94` |
| OCR 漏字、错字多 | 分辨率太低 | 提高 `--band` 精度；或用 `--ocr vlm` 对比 |

调参顺序：**先 `--band`，再 `--sim`，最后 `--ink`**。前两个解决 90% 的问题。

## 翻译后端

默认用 B 站开源的 Index-Translate 公开端点，**免费、不用 key**，官方限非商用、无 SLA。

```bat
:: 默认（免费）
jpsub.cmd video.mp4 --band 0.81,0.97

:: 换自己的 API（更稳、可商用）
jpsub.cmd video.mp4 --band 0.81,0.97 --translate openai ^
    --base-url https://api.deepseek.com/v1 --api-key sk-xxx --model deepseek-chat
```

`--translate openai` 只要目标服务兼容 OpenAI 协议就行：vLLM 自建的 Mango-MT-9B、DeepSeek、硅基流动、官方 OpenAI 都能用。

## 自动填充原字幕颜色（不同说话人不同颜色）

颜色识别是默认行为，不需要额外开关：

```bat
jpsub.cmd video.mp4 --band 0.80,0.94
```

- 从画面里识别每条字幕的**填充色**（区分开描边，黑字/粉字这种带白描边的也能认）
- 全片颜色自动归并成几种，同一个人前后一致
- 直接写进 `.ass`；黑字会自动配白描边
- VLC / PotPlayer / ffmpeg 压片都认 `.ass`，不需要再手动套样式

## 已有日文 SRT，只翻译

```bat
jpsub.cmd srt "D:\videos\某番剧.ja.srt" --translate openai --base-url ... --api-key ...
```

输出同名的中文 `.ass`（已有日文 SRT 没有画面，因此不带颜色）。

## OCR 后端

| 后端 | 说明 |
|---|---|
| `paddle`（默认） | PP-OCRv6 日文模型（PaddleOCR 3.7 默认），本地推理，逐字最准 |
| `vlm` | B 站免费视觉接口，零安装，但会顺手改写原文（`雨が降る`→`雨の降る`），只作对比 |

两者对不上时，以 `paddle` 为准。`vlm` 的价值是当 paddle 装不上时的兜底。

## 全部参数

```
jpsub.cmd run <视频> [options]

  --band a,b          字幕带上下边界（相对高度），默认 0.70,0.99；写 auto 自动猜
  --sub-color C       切句用的笔画颜色：auto/white/yellow/black，默认 auto
  --fps N             变化检测采样帧率，默认 6（时间边界误差 ≈ 1/N 秒）
  --sim F             切换阈值，越小越敏感，默认 0.62
  --ink N             笔画二值化阈值，默认 34
  --min-dur S         最短字幕时长，默认 0.24
  --confirm N         连续多少帧判定换句，默认 4（约 0.67s，抗字幕抖动）
  --max-seconds S     只处理前 N 秒
  --debug-dir DIR     cue 截图目录，默认 <输出目录>/cues
  -o, --out DIR       输出目录

  --ocr paddle|vlm    OCR 后端
  --translate bilibili|openai|none
  --model ID          翻译模型
  --base-url URL      openai 后端地址
  --api-key KEY       openai 后端 key（或环境变量 OPENAI_API_KEY）
  --batch N           每批条数，默认 16
  --concurrency N     并发批数，默认 2
  --max-line N        单行超多少字断行，默认 20，0 为不断
```

## 目录结构

```
D:\ai\jpsub\
  jpsub.cmd              命令行入口
  jpsub-gui.cmd          图形界面入口
  setup.cmd              一次性装环境
  requirements.txt
  .venv\                 隔离环境
  jpsub\
    cli.py               命令行
    gui.py               图形界面（Tkinter）
    locate.py            字幕定位测试（抓几帧拼图）
    pipeline.py          主流程编排
    video.py             探测 + 抽帧（ffmpeg / cv2）
    cues.py              字幕变化检测
    band.py              字幕带自动定位
    ocr.py               OCR 后端
    translate.py         翻译后端
    srtio.py             读 SRT / 写 ASS
```

## 它做不到的

- **竖排字幕**：日式综艺的竖排字幕，PP-OCRv5 会打乱顺序，需人工核对
- **手写字、艺术字、拟声词**：基本无解
- **双语同屏（日文+英文同时出现）**：会把两行一起识别，翻译结果混在一起
- **时间轴精度**：受 `--fps` 限制，默认 6fps 误差 ±0.17 秒。要更准就 `--fps 12`（慢一倍）

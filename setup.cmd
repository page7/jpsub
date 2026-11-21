@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   jpsub 环境初始化
echo ============================================
echo.

rem ---- 1. 挑一个 Python（PaddlePaddle 对 3.13 支持不稳，优先 3.12）
set "BASEPY="
for %%V in (3.12 3.11 3.13) do (
  if not defined BASEPY (
    py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
  )
)
if not defined BASEPY (
  python -c "import sys; assert sys.version_info[:2] >= (3,10)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
  echo [x] 没找到可用的 Python（需要 3.10 以上）。请先装 Python 3.12。
  exit /b 1
)
echo [1/3] 使用解释器: !BASEPY!
!BASEPY! -c "import sys; print('      ', sys.version)"

rem ---- 2. 建虚拟环境
if not exist ".venv\Scripts\python.exe" (
  echo [2/3] 创建虚拟环境 .venv ...
  !BASEPY! -m venv .venv
  if errorlevel 1 (
    echo [x] 创建虚拟环境失败
    exit /b 1
  )
) else (
  echo [2/3] 虚拟环境已存在，跳过
)

set "PY=%CD%\.venv\Scripts\python.exe"

rem ---- 3. 装依赖
echo [3/3] 安装依赖（首次约 400MB，慢）
"%PY%" -m pip install -q --upgrade pip
"%PY%" -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo [x] 依赖安装失败。可尝试换源后重跑:
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  exit /b 1
)

echo.
echo 验证安装 ...
"%PY%" -c "import cv2, numpy, requests; print('  opencv', cv2.__version__); print('  numpy ', numpy.__version__)"
"%PY%" -c "import paddleocr; print('  paddleocr ok')" || echo "  [!] paddleocr 导入失败，OCR 可改用 --ocr vlm"
"%PY%" -c "import imageio_ffmpeg as f; print('  ffmpeg ', f.get_ffmpeg_exe())"

echo.
echo ============================================
echo   完成。用法:
echo     jpsub.cmd probe "D:\videos\某番剧.mp4"
echo     jpsub.cmd "D:\videos\某番剧.mp4"
echo ============================================

@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [x] 还没初始化环境，请先运行 setup.cmd
  pause
  exit /b 1
)
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
start "" "%PY%" -m jpsub.gui
exit /b 0

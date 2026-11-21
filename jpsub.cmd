@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [x] 还没初始化环境，请先运行 setup.cmd
  exit /b 1
)
rem 不切换工作目录，这样命令里的相对路径仍按你当前所在的目录解析
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
"%PY%" -m jpsub %*
exit /b %errorlevel%

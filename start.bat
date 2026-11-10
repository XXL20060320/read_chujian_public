@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 简牍识字 · 启动器

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   简牍识字 · 楚简字形识别与形近字检索
echo ============================================================
echo.

if not exist "app.py" goto :no_files
if not exist "bundle_cnn\model.pt" goto :no_model
if not exist "bundle_cnn\X_train.npy" goto :no_cache

set "PY="
call :probe "%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe"
if not defined PY call :probe "%LOCALAPPDATA%\Python\bin\python.exe"
if not defined PY call :probe "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY call :probe "C:\Python311\python.exe"
if not defined PY for /f "delims=" %%p in ('where python 2^>nul') do if not defined PY call :probe "%%p"
if not defined PY goto :no_python

:launch
echo 解释器: %PY%
echo 正在启动，浏览器会自动打开 http://127.0.0.1:7860
echo 用完关闭本窗口即可停止服务。
echo.
"%PY%" app.py --bundle bundle_cnn %*
if errorlevel 1 goto :failed
goto :eof

:probe
if not exist "%~1" exit /b
"%~1" -c "import torch, gradio, numpy, pandas, PIL" >nul 2>&1
if errorlevel 1 exit /b
set "PY=%~1"
exit /b

:no_python
echo [错误] 没有找到已安装 torch 与 gradio 的 Python。
echo.
echo 请先安装依赖，然后重新双击本文件：
echo     python -m pip install -r requirements.txt
echo.
pause
exit /b 1

:no_files
echo [错误] 当前目录缺少 app.py，请确认本文件与程序放在同一文件夹内。
pause
exit /b 1

:no_model
echo [错误] 缺少 bundle_cnn\model.pt，模型包不完整。
pause
exit /b 1

:no_cache
echo [错误] 缺少 bundle_cnn\X_train.npy，界面无法显示形近字缩略图。
pause
exit /b 1

:failed
echo.
echo [错误] 程序异常退出，请把上面的报错信息发给我。
pause
exit /b 1

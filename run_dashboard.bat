@echo off
set PYTHON_EXE=C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe

if not exist "%PYTHON_EXE%" (
  echo Python runtime not found at %PYTHON_EXE%
  exit /b 1
)

cd /d "%~dp0"
"%PYTHON_EXE%" server.py

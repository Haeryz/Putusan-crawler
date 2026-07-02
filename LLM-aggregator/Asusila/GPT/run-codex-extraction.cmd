@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-codex-extraction.ps1" %*
exit /b %ERRORLEVEL%

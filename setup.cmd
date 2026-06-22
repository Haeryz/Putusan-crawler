@echo off
REM Double-click / one-command bootstrap for Windows.
REM Usage: setup.cmd [target-number]   e.g.  setup.cmd 20
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
exit /b %ERRORLEVEL%

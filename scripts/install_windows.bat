@echo off
rem Double-click to install everything GLaDOS needs on Windows 10/11.
rem Options are passed to install_windows.ps1, for example:
rem   scripts\install_windows.bat -Backend cpu -NoOllama
rem   scripts\install_windows.bat -Api -OllamaModels gemma4:e4b
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" echo Installation failed with exit code %CODE%.
pause
exit /b %CODE%

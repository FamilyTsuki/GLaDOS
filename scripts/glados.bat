@echo off
rem Windows launcher for the default GLaDOS profile.
rem
rem   scripts\glados.bat          start GLaDOS if stopped, stop it if running
rem   scripts\glados.bat start    start the model server and the web console
rem   scripts\glados.bat stop     stop both
rem
rem LLAMA_SERVER     path to a Gemma 4-capable CUDA llama-server.exe (default: llama-server on PATH)
rem GLADOS_MODEL_DIR directory holding the Gemma 4 GGUF files (default: models\gemma4-benchmark)
rem GLADOS_CONFIG    web console configuration (default: configs\glados_webapp_config.yaml)
rem
rem Server options mirror scripts/run_llamacpp.sh; keep the two in sync.
setlocal EnableExtensions
cd /d "%~dp0.."

set "SERVER_TITLE=GLaDOS llama-server"
set "CONSOLE_TITLE=GLaDOS web console"
set "HEALTH_URL=http://127.0.0.1:18080/health"
set "WEB_URL=http://127.0.0.1:8050/"
if not defined LLAMA_SERVER set "LLAMA_SERVER=llama-server"
if not defined GLADOS_MODEL_DIR set "GLADOS_MODEL_DIR=%CD%\models\gemma4-benchmark"
if not defined GLADOS_CONFIG set "GLADOS_CONFIG=configs\glados_webapp_config.yaml"

if /i "%~1"=="start" goto start
if /i "%~1"=="stop" goto stop
if not "%~1"=="" (
    echo Usage: %~nx0 [start^|stop]
    exit /b 2
)
tasklist /v /fi "imagename eq cmd.exe" /fo csv | find "%CONSOLE_TITLE%" >nul && goto stop
tasklist /v /fi "imagename eq llama-server.exe" /fo csv | find "%SERVER_TITLE%" >nul && goto stop

:start
where "%LLAMA_SERVER%" >nul 2>&1 || if not exist "%LLAMA_SERVER%" (
    echo Set LLAMA_SERVER to a Gemma 4-capable CUDA llama-server executable.
    exit /b 1
)
for %%m in (gemma-4-E4B-it-Q4_0.gguf mmproj-gemma-4-E4B-it-Q8_0.gguf) do if not exist "%GLADOS_MODEL_DIR%\%%m" (
    echo Missing model: %GLADOS_MODEL_DIR%\%%m ^(see docs\gemma4.md^).
    exit /b 1
)

curl -s -m 2 %HEALTH_URL% 2>nul | find "ok" >nul && goto console
echo Starting llama-server...
start "%SERVER_TITLE%" /min "%LLAMA_SERVER%" ^
    -m "%GLADOS_MODEL_DIR%\gemma-4-E4B-it-Q4_0.gguf" ^
    --mmproj "%GLADOS_MODEL_DIR%\mmproj-gemma-4-E4B-it-Q8_0.gguf" ^
    --alias gemma-4-E4B --host 127.0.0.1 --port 18080 ^
    -c 65536 -ngl all --parallel 4 --cont-batching --reasoning off ^
    --swa-full --cache-type-k q8_0 --cache-type-v q8_0 ^
    --flash-attn on --cache-prompt --cache-ram 8192 ^
    --no-cache-idle-slots -b 2048 -ub 512 ^
    --log-verbosity 2

set /a TRIES=0
:wait_server
timeout /t 2 /nobreak >nul
curl -s -m 2 %HEALTH_URL% 2>nul | find "ok" >nul && goto console
set /a TRIES+=1
if %TRIES% lss 90 goto wait_server
echo llama-server did not become ready; check the "%SERVER_TITLE%" window.
exit /b 1

:console
echo Starting the GLaDOS web console...
start "%CONSOLE_TITLE%" /min cmd /c uv run --no-sync glados webapp --config "%GLADOS_CONFIG%"

set /a TRIES=0
:wait_console
timeout /t 2 /nobreak >nul
curl -s -m 2 -o nul %WEB_URL% 2>nul && goto open
set /a TRIES+=1
if %TRIES% lss 90 goto wait_console
echo The web console did not start; check the "%CONSOLE_TITLE%" window.
exit /b 1

:open
start "" %WEB_URL%
exit /b 0

:stop
echo Stopping GLaDOS...
taskkill /fi "windowtitle eq %CONSOLE_TITLE%*" /t /f >nul 2>&1
taskkill /fi "windowtitle eq %SERVER_TITLE%*" /t /f >nul 2>&1
set /a TRIES=0
:wait_stop
curl -s -m 1 -o nul %WEB_URL% 2>nul || curl -s -m 1 -o nul %HEALTH_URL% 2>nul || goto stopped
timeout /t 1 /nobreak >nul
set /a TRIES+=1
if %TRIES% lss 20 goto wait_stop
:stopped
echo GLaDOS stopped.
exit /b 0

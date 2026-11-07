<#
.SYNOPSIS
    Installs everything GLaDOS needs on Windows 10/11.

.DESCRIPTION
    1. Installs Git, uv, the Visual C++ Redistributable and Ollama with winget.
    2. Runs scripts\install.py through uv (Python 3.12 environment, ONNX runtime,
       speech models: Parakeet, Silero VAD, English and French GLaDOS voices).
    3. Starts Ollama if needed and pulls the conversation and vision models.

    Safe to run again: anything already installed is skipped, and install.py
    keeps the existing environment.

.EXAMPLE
    scripts\install_windows.bat
.EXAMPLE
    scripts\install_windows.bat -Backend cpu -NoOllama
.EXAMPLE
    scripts\install_windows.bat -OllamaModels gemma4:e4b
#>
[CmdletBinding()]
param(
    # ONNX runtime for the speech models: auto picks cuda when nvidia-smi exists, cpu otherwise.
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Backend = 'auto',
    # Also install the HTTP API dependencies (litestar).
    [switch]$Api,
    # Skip Ollama and the model downloads (for an OpenAI-compatible API or llama-server).
    [switch]$NoOllama,
    # Ollama models to pull; the defaults are the ones configs\my_glados.yaml uses.
    [string[]]$OllamaModels = @('qwen3:4b-instruct', 'qwen2.5vl:3b')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "    $Message" -ForegroundColor Green }

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Update-SessionPath {
    # winget installers only update the registry; reload PATH so the new tools work in this session.
    $paths = @(
        [Environment]::GetEnvironmentVariable('Path', 'Machine'),
        [Environment]::GetEnvironmentVariable('Path', 'User'),
        "$env:USERPROFILE\.local\bin",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
        "$env:LOCALAPPDATA\Programs\Ollama"
    )
    $env:Path = ($paths | Where-Object { $_ }) -join ';'
}

function Install-WingetPackage([string]$Id, [string]$Label, [scriptblock]$IsInstalled) {
    if (& $IsInstalled) {
        Write-Ok "$Label already installed"
        return
    }
    Write-Host "    Installing $Label ($Id)..."
    winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements
    $code = $LASTEXITCODE
    Update-SessionPath
    if (-not (& $IsInstalled)) {
        throw "$Label installation failed (winget exit code $code). Install it manually, then run this script again."
    }
    Write-Ok "$Label installed"
}

function Test-VCRedist {
    foreach ($key in @(
            'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
            'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
        )) {
        $entry = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
        if ($entry -and $entry.Installed -eq 1) { return $true }
    }
    return $false
}

function Test-OllamaServer {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Invoke-Checked([string]$What, [string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)." }
}

Write-Host 'GLaDOS installer for Windows' -ForegroundColor Yellow
Write-Host "Project: $Root"

if ([Environment]::Is64BitOperatingSystem -eq $false) {
    throw 'GLaDOS needs 64-bit Windows.'
}

Write-Step 'Checking winget'
Update-SessionPath
if (-not (Test-Command 'winget')) {
    throw ('winget is missing. Install "App Installer" from the Microsoft Store ' +
        '(https://apps.microsoft.com/detail/9NBLGGH4NNS1), then run this script again.')
}
Write-Ok 'winget found'

Write-Step 'System dependencies'
Install-WingetPackage 'Git.Git' 'Git' { Test-Command 'git' }
Install-WingetPackage 'astral-sh.uv' 'uv' { Test-Command 'uv' }
Install-WingetPackage 'Microsoft.VCRedist.2015+.x64' 'Visual C++ Redistributable' { Test-VCRedist }
# The French GLaDOS voice phonemizes text with espeak-ng.
Install-WingetPackage 'eSpeak-NG.eSpeak-NG' 'eSpeak NG' {
    (Test-Command 'espeak-ng') -or (Test-Path "$env:ProgramFiles\eSpeak NG\espeak-ng.exe")
}
if (-not $NoOllama) {
    Install-WingetPackage 'Ollama.Ollama' 'Ollama' { Test-Command 'ollama' }
}

Write-Step "Python environment, ONNX runtime ($Backend) and speech models"
# uv downloads Python 3.12 itself: no system Python needed.
# --isolated keeps uv from running install.py with .venv\Scripts\python.exe,
# which would lock the file install.py needs to recreate (os error 32).
$installArgs = @('run', '--no-project', '--isolated', '--python', '3.12', 'scripts\install.py', '--backend', $Backend)
if ($Api) { $installArgs += '--api' }
Invoke-Checked 'scripts\install.py' 'uv' $installArgs
Write-Ok 'Python environment ready'

if (-not $NoOllama) {
    Write-Step 'Ollama models'
    if (-not (Test-OllamaServer)) {
        Write-Host '    Starting the Ollama server...'
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
        $deadline = (Get-Date).AddSeconds(60)
        while (-not (Test-OllamaServer)) {
            if ((Get-Date) -gt $deadline) { throw 'The Ollama server did not start on http://127.0.0.1:11434.' }
            Start-Sleep -Seconds 2
        }
    }
    foreach ($model in $OllamaModels) {
        Write-Host "    Pulling $model..."
        Invoke-Checked "ollama pull $model" 'ollama' @('pull', $model)
    }
    Write-Ok 'Models ready'
}

if (-not (Test-Path '.env')) {
    Write-Host "`n    Note: no .env file. Add OPENAI_API_KEY=... there for the API worker and Nexus." -ForegroundColor DarkYellow
}

Write-Host "`nGLaDOS is installed." -ForegroundColor Green
Write-Host 'Start the web console with:'
Write-Host '    uv run --no-sync glados webapp --config configs\glados_extended_config.yaml --config configs\my_glados.yaml'
Write-Host 'then open http://127.0.0.1:8050/'
Write-Host '(Open a new terminal first so uv and ollama are on PATH.)'

<#
.SYNOPSIS
    Turns a Windows PC into a second home for your GLaDOS, kept in step with the first one.

.DESCRIPTION
    Run it from the cloned repository (branch myGLaDOS), after the shared folder has received the
    state from the first PC (there: uv run glados sync <shared folder>).

    1. Installs GLaDOS with scripts\install_windows.ps1 (uv, Ollama, speech models, chat + vision models).
    2. Runs a first sync (the folder is remembered for every later start and stop): settings, memory,
       phone pairing, API keys and the fine-tuned model arrive from the other PC.
    3. Imports the fine-tuned model into Ollama as glados-ft.
    4. Lets the phone reach the console (8050) and the coding previews (8100-8199) on private networks
       (one administrator prompt).

    Safe to run again.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_pc.ps1 -SyncDir "$HOME\Sync\GLaDOS"
#>
[CmdletBinding()]
param(
    # The folder shared with the other PC (Syncthing, OneDrive, a USB stick...).
    [Parameter(Mandatory = $true)]
    [string]$SyncDir,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }

if (-not (Test-Path $SyncDir)) { throw "Shared folder not found: $SyncDir" }
$SyncDir = (Resolve-Path $SyncDir).Path

if (-not $SkipInstall) {
    Write-Step 'Installing GLaDOS (uv, Ollama, speech and language models)'
    & "$PSScriptRoot\install_windows.ps1" -OllamaModels @('qwen3:4b-instruct', 'qwen2.5vl:7b')
}

Write-Step 'First sync with the other PC (this PC then syncs on every start and stop)'
& uv run --no-sync glados sync "$SyncDir"
if ($LASTEXITCODE -ne 0) { throw 'glados sync failed' }

Write-Step 'Fine-tuned model'
if (Test-Path (Join-Path $Root 'training\glados-q4_k_m.gguf')) {
    Push-Location (Join-Path $Root 'training')
    & ollama create glados-ft -f Modelfile
    Pop-Location
} else {
    Write-Warning 'training\glados-q4_k_m.gguf did not arrive yet; GLaDOS falls back to nothing until it does. Re-run once the shared folder has it.'
}

Write-Step 'Firewall: console and previews reachable from the phone (private networks only)'
$rules = @(
    'netsh advfirewall firewall add rule name="GLaDOS console" dir=in action=allow protocol=TCP localport=8050 profile=private',
    'netsh advfirewall firewall add rule name="GLaDOS previews" dir=in action=allow protocol=TCP localport=8100-8199 profile=private'
)
& netsh advfirewall firewall show rule name="GLaDOS previews" *> $null
if ($LASTEXITCODE -ne 0) {
    Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile', '-Command', ($rules -join '; ')
}

Write-Step 'Done. Still to do by hand (once):'
Write-Host @'
  - Git for Windows (Claude Code and the previews need its bash): winget install Git.Git
  - Claude Code, then sign in: irm https://claude.ai/install.ps1 | iex ; claude
  - Antigravity CLI (agy) if you use it, then sign in
  - GitHub CLI: winget install GitHub.cli ; gh auth login
  - KDE Connect (previews open on the phone): winget install KDE.KDEConnect, then pair the phone
  - Start GLaDOS: uv run --no-sync glados webapp --config configs\glados_extended_config.yaml --config configs\my_glados.yaml
    (configs\machines\win32.yaml is added automatically; sync runs before start and after stop)
'@

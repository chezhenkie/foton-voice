<#
.SYNOPSIS
  Prepare one STT engine scenario for a clean test run, then report the results.

.DESCRIPTION
  Rotates startup_errors.log so each scenario stands alone, and can hide or
  restore onnxruntime_providers_webgpu.dll to force Moonshine onto the CPU.
  Switch the engine itself in the app's Settings; this script never edits
  config.json, because that file carries duplicate VoxCtrl/voxctrl keys and
  rewriting it by regex corrupted it during testing.

  Why the dll rename is the Moonshine CPU lever: Moonshine has no gpu flag and
  moonshine_gpu_backend() is decided at compile time, so it always claims
  "webgpu". The real gate is webgpu_devices(), which returns empty when the
  provider dll is absent, and moonshine.rs then builds a plain CPU session.
  Nemotron does have a gpu flag, so switch that in the UI.

.EXAMPLE
  .\stt_test.ps1 -Start moonshine
.EXAMPLE
  .\stt_test.ps1 -Start moonshine -Cpu
.EXAMPLE
  .\stt_test.ps1 -Report
#>
param(
  [ValidateSet('whisper', 'moonshine', 'nemotron')]
  [string]$Engine,

  [switch]$Cpu,

  [switch]$Start,

  [switch]$Report,

  [string]$Log        = "C:\apps\VoxCtrl\startup_errors.log",
  [string]$InstallDir = "C:\apps\VoxCtrl",
  [string]$ArchiveDir = ""
)

$ErrorActionPreference = 'Stop'

if (-not $ArchiveDir) {
  $ArchiveDir = Join-Path $PSScriptRoot "..\..\..\.stt-logs"
}

function Assert-AppClosed {
  $p = Get-Process -Name 'fotonvoice-engine' -ErrorAction SilentlyContinue
  if ($p) {
    throw "fotonvoice-engine is running (PID $($p.Id -join ',')). Quit it: config and log are read at startup."
  }
}

function Invoke-Start {
  Assert-AppClosed
  $scenario = $Engine + $(if ($Cpu) { '-cpu' } else { '-gpu' })

  if (Test-Path -LiteralPath $Log) {
    New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
    $dest = Join-Path $ArchiveDir ((Get-Date -Format 'yyyyMMdd-HHmmss') + "-$scenario.log")
    Move-Item -LiteralPath $Log -Destination $dest -Force
    Write-Host "archived previous log -> $dest" -ForegroundColor DarkGray
  }

  $dll = Join-Path $InstallDir 'onnxruntime_providers_webgpu.dll'
  $off = "$dll.off"

  if ($Cpu) {
    if (Test-Path -LiteralPath $off) {
      Move-Item -LiteralPath $off -Destination $dll -Force
      Write-Host "restored webgpu provider dll (was already hidden)" -ForegroundColor DarkGray
    }
    if (Test-Path -LiteralPath $dll) {
      Move-Item -LiteralPath $dll -Destination $off -Force
      Write-Host "webgpu provider dll HIDDEN -> Moonshine will use the CPU" -ForegroundColor Yellow
    } else {
      Write-Host "webgpu provider dll already hidden" -ForegroundColor DarkGray
    }
  } else {
    if (Test-Path -LiteralPath $off) {
      Move-Item -LiteralPath $off -Destination $dll -Force
      Write-Host "webgpu provider dll RESTORED -> GPU path available" -ForegroundColor Yellow
    } else {
      Write-Host "webgpu provider dll in place" -ForegroundColor DarkGray
    }
  }

  Write-Host ""
  Write-Host "scenario $scenario ready." -ForegroundColor Green
  if ($Engine -eq 'nemotron' -or $Engine -eq 'moonshine') {
    Write-Host "1. set the engine to $Engine in Settings" -ForegroundColor Yellow
  }
  if ($Engine -eq 'nemotron') {
    Write-Host "2. set Nemotron GPU to $(if ($Cpu) { 'off' } else { 'ON' }) in Settings" -ForegroundColor Yellow
  }
  Write-Host "3. start fotonvoice-engine, record ONE clip, then quit" -ForegroundColor Yellow
  Write-Host "4. run: .\stt_test.ps1 -Report" -ForegroundColor Yellow
}

function Show-Results {
  param([string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    Write-Host "no log at $Path - nothing logged yet" -ForegroundColor Yellow
    return
  }

  $lines = [System.IO.File]::ReadAllLines($Path)
  Write-Host "log: $Path  ($($lines.Count) lines)" -ForegroundColor DarkGray
  Write-Host ""

  $rx = [regex]'(Moonshine acceleration|Moonshine geometry|Moonshine: .*execution provider|Nemotron streaming: .*(attached|unavailable)|Whisper acceleration|Whisper run:|WebGPU: |Failed to load|still not loadable|Loading Moonshine|Loading Nemotron|Backend choice)'
  $hits = @($lines | Where-Object { $_ -match $rx })

  if ($hits.Count -eq 0) {
    Write-Host "nothing matched. Last 15 lines:" -ForegroundColor Yellow
    $lines | Select-Object -Last 15 | ForEach-Object { Write-Host "  $_" }
    return
  }
  foreach ($h in $hits) { Write-Host ("  " + ($h -replace '^\S+ \S+ ', '')) }

  if (-not ($lines | Where-Object { $_ -match 'Whisper run:' })) {
    Write-Host ""
    Write-Host "no timing line: Moonshine and Nemotron compute inference_ms but never log it." -ForegroundColor Yellow
    Write-Host "only Whisper emits a 'Whisper run:' line." -ForegroundColor Yellow
  }
}

if ($Start) {
  if (-not $Engine) { throw "-Start needs -Engine whisper|moonshine|nemotron" }
  Invoke-Start
}
if ($Report) { Show-Results -Path $Log }
if (-not $Start -and -not $Report) {
  Write-Host "give -Start <engine> [-Cpu] and/or -Report" -ForegroundColor Yellow
  Get-Help $PSCommandPath -Detailed
}
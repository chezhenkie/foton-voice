<#
.SYNOPSIS
  Flip the Whisper compute device and report inference timings.

.DESCRIPTION
  Two jobs, used together for a CPU vs GPU comparison on identical audio:

    1. Set the device, then start the app and dictate the same clip once.
       The app reads config.json at startup, so it must be restarted.
         .\bench_whisper.ps1 -Device vulkan
         .\bench_whisper.ps1 -Device cpu
    2. Read the numbers the app already logged.
         .\bench_whisper.ps1 -Report

  The report is parsed from startup_errors.log, not measured here, so the
  numbers are the ones the process actually produced. Each row shows the
  real-time factor: audio_ms / inference_ms, so 12.0 means it transcribed 12
  seconds of audio per second of wall clock. Anything below 1.0 is slower than
  real time.

.EXAMPLE
  .\bench_whisper.ps1 -Device vulkan
.EXAMPLE
  .\bench_whisper.ps1 -Report
#>
param(
  [ValidateSet('auto', 'vulkan', 'cpu', 'cuda')]
  [string]$Device,

  [switch]$Report,

  [string]$Config = "C:\apps\VoxCtrl\config.json",
  [string]$Log    = "C:\apps\VoxCtrl\startup_errors.log"
)

$ErrorActionPreference = 'Stop'

function Set-WhisperDevice {
  param([string]$Path, [string]$Value)

  if (-not (Test-Path -LiteralPath $Path)) {
    throw "config not found: $Path"
  }

  # Back up once, so a bad edit is always recoverable.
  $bak = "$Path.benchbak"
  if (-not (Test-Path -LiteralPath $bak)) {
    Copy-Item -LiteralPath $Path -Destination $bak
    Write-Host "backed up config -> $bak" -ForegroundColor DarkGray
  }

  $text = [System.IO.File]::ReadAllText($Path)

  # Edit the text, never the parsed object: this config contains both "VoxCtrl"
  # and "voxctrl" keys, and round-tripping it through a JSON parser would drop
  # one of them. Only the single "device" key inside whisper_cpp is touched.
  $rx = [regex]'"device"\s*:\s*"[^"]*"'
  $matches = $rx.Matches($text)
  if ($matches.Count -ne 1) {
    throw "expected exactly 1 device key, found $($matches.Count). Refusing to guess."
  }

  $updated = $rx.Replace($text, '"device": "' + $Value + '"', 1)
  [System.IO.File]::WriteAllText($Path, $updated)

  Write-Host "device = $Value" -ForegroundColor Green
  Write-Host "restart the app before recording, it reads config.json at startup" -ForegroundColor Yellow
}

function Get-WhisperReport {
  param([string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    throw "log not found: $Path"
  }

  $runRx = [regex]'Whisper run: backend = (?<backend>[^,]+), device = (?<device>[^,]+), threads = (?<threads>[^,]+), audio = (?<audio>\d+) ms, inference = (?<infer>\d+) ms'

  $accel = @()
  $runs  = @()
  foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
    if ($line -match 'Whisper acceleration: (?<a>.+)$') {
      $accel += $Matches['a'].Trim()
    }
    if ($line -match $runRx) {
      $runs += [pscustomobject]@{
        Time     = ($line -split '\s+')[0]
        Backend  = $Matches['backend'].Trim()
        Device   = $Matches['device'].Trim()
        Threads  = $Matches['threads'].Trim()
        AudioMs  = [int]$Matches['audio']
        InferMs  = [int]$Matches['infer']
      }
    }
  }

  Write-Host "log: $Path" -ForegroundColor DarkGray
  if ($accel.Count -gt 0) {
    Write-Host ""
    Write-Host "acceleration reported at startup:"
    $accel | Select-Object -Unique | ForEach-Object { Write-Host "  $_" }
  }

  Write-Host ""
  if ($runs.Count -eq 0) {
    Write-Host "no 'Whisper run:' lines yet. Record something first." -ForegroundColor Yellow
    return
  }

  Write-Host ("{0,-22} {1,-8} {2,-8} {3,-8} {4,10} {5,10} {6,8}" -f `
    'timestamp', 'backend', 'device', 'threads', 'audio_ms', 'infer_ms', 'rtf')
  Write-Host ("-" * 82)
  foreach ($r in $runs) {
    $rtf = if ($r.InferMs -gt 0) { [math]::Round($r.AudioMs / $r.InferMs, 2) } else { 0 }
    Write-Host ("{0,-22} {1,-8} {2,-8} {3,-8} {4,10} {5,10} {6,8}" -f `
      $r.Time, $r.Backend, $r.Device, $r.Threads, $r.AudioMs, $r.InferMs, $rtf)
  }

  # Per-device mean, which is the number that decides CPU vs GPU.
  Write-Host ""
  Write-Host "mean real-time factor by device:"
  $runs | Group-Object Device | ForEach-Object {
    $audio = ($_.Group | Measure-Object -Property AudioMs -Sum).Sum
    $infer = ($_.Group | Measure-Object -Property InferMs -Sum).Sum
    $rtf = if ($infer -gt 0) { [math]::Round($audio / $infer, 2) } else { 0 }
    Write-Host ("  {0,-8} n={1,-4} audio={2,7} ms  infer={3,8} ms  rtf={4}" -f `
      $_.Name, $_.Count, $audio, $infer, $rtf)
  }
  Write-Host ""
  Write-Host "Compare like for like: same clip, same model, same thread count." -ForegroundColor DarkGray
}

if ($Device) { Set-WhisperDevice -Path $Config -Value $Device }
if ($Report) { Get-WhisperReport -Path $Log }
if (-not $Device -and -not $Report) {
  Write-Host "give -Device <auto|vulkan|cpu|cuda> and/or -Report" -ForegroundColor Yellow
  Get-Help $PSCommandPath -Detailed
}
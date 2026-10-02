#Requires -Version 5.1
<#
.SYNOPSIS
    Build FotonVoice Engine for Windows.

.PARAMETER Cuda
    Enable CUDA GPU acceleration (requires NVIDIA GPU + CUDA Toolkit).

.PARAMETER Vulkan
    Enable Vulkan GPU acceleration for whisper.cpp (works on NVIDIA, AMD and
    Intel GPUs; runtime needs only the GPU driver's Vulkan loader). Build needs
    the LunarG Vulkan SDK with VULKAN_SDK set, or a GitHub CI build.

.PARAMETER Debug
    Build in debug mode (faster compile, larger binary, no optimisation).

.PARAMETER SkipNodeInstall
    Skip 'npm install' (use when deps are already installed).

.EXAMPLE
    .\build_windows.ps1
    .\build_windows.ps1 -Cuda
    .\build_windows.ps1 -Vulkan
    .\build_windows.ps1 -Debug
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$Vulkan,
    [switch]$Debug,
    [switch]$SkipNodeInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
    Write-Host "    [OK] $msg" -ForegroundColor Green
}

function Write-Fail([string]$msg) {
    Write-Host "    [FAIL] $msg" -ForegroundColor Red
    exit 1
}

function Require-Command([string]$cmd, [string]$hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Fail "$cmd not found. $hint"
    }
    Write-Ok "$cmd found"
}

# ── Prerequisite checks ───────────────────────────────────────────────────────

Write-Step "Checking prerequisites"

Require-Command "cargo"  "Install Rust from https://rustup.rs/"
Require-Command "node"   "Install Node.js 18+ from https://nodejs.org/"
Require-Command "npm"    "Install Node.js 18+ from https://nodejs.org/"

# Verify MSVC toolchain
$rustTarget = (rustup show active-toolchain 2>$null) -replace '\s.*',''
if ($rustTarget -notlike "*windows-msvc*") {
    Write-Host "    [WARN] Active toolchain '$rustTarget' is not MSVC." -ForegroundColor Yellow
    Write-Host "           Run: rustup default stable-x86_64-pc-windows-msvc" -ForegroundColor Yellow
}
Write-Ok "Rust toolchain: $rustTarget"

# Verify tauri-cli
if (-not (Get-Command "cargo-tauri" -ErrorAction SilentlyContinue)) {
    Write-Host "    tauri-cli not found — installing..." -ForegroundColor Yellow
    cargo install tauri-cli
}
Write-Ok "tauri-cli found"

# CUDA check
if ($Cuda) {
    if (-not $env:CUDA_PATH) {
        $cudaDirs = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" `
            -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        if ($cudaDirs) {
            $env:CUDA_PATH = $cudaDirs[0].FullName
            Write-Host "    Auto-detected CUDA at: $($env:CUDA_PATH)" -ForegroundColor Yellow
        } else {
            Write-Fail "CUDA_PATH not set and no CUDA Toolkit found. Install from https://developer.nvidia.com/cuda-downloads"
        }
    }
    Write-Ok "CUDA_PATH = $env:CUDA_PATH"
}

# Vulkan check
if ($Vulkan) {
    if (-not $env:VULKAN_SDK -and (Test-Path "C:\VulkanSDK")) {
        $sdkDirs = Get-ChildItem "C:\VulkanSDK" -Directory | Sort-Object Name -Descending
        if ($sdkDirs) {
            $env:VULKAN_SDK = $sdkDirs[0].FullName
            Write-Host "    Auto-detected Vulkan SDK at: $($env:VULKAN_SDK)" -ForegroundColor Yellow
        }
    }
    if (-not $env:VULKAN_SDK -or -not (Test-Path "$env:VULKAN_SDK\Include") -or -not (Test-Path "$env:VULKAN_SDK\Lib")) {
        Write-Fail "VULKAN_SDK not set (or missing Include/Lib). Install the LunarG Vulkan SDK or build via GitHub CI (.github/workflows/build-msvc.yml, build-vulkan job)."
    }
    if (-not (Test-Path "$env:VULKAN_SDK\Bin\glslc.exe")) {
        Write-Fail "glslc.exe not found at $env:VULKAN_SDK\Bin\glslc.exe. ggml-vulkan compiles its shaders at build time."
    }
    Write-Ok "VULKAN_SDK = $env:VULKAN_SDK"
}

# ── Frontend ──────────────────────────────────────────────────────────────────

if (-not $SkipNodeInstall) {
    Write-Step "Installing frontend dependencies"
    npm install
    if ($LASTEXITCODE -ne 0) { Write-Fail "npm install failed" }
    Write-Ok "npm install complete"
}

# ── Build ─────────────────────────────────────────────────────────────────────

Write-Step "Building FotonVoice Engine"

# THE fix for STATUS_ILLEGAL_INSTRUCTION (0xc000001d) on AVX-512-less CPUs such
# as Intel Lunar Lake, done WITHOUT giving up CPU speed.
#
# ggml picks its x86 kernel set from two variables (ggml/CMakeLists.txt):
#     if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT) -> INS_ENB OFF else INS_ENB ON
# and GGML_SSE42, GGML_AVX, GGML_AVX2, GGML_BMI2, GGML_FMA, GGML_F16C all
# default to INS_ENB. Only GGML_NATIVE=ON passes -march=native, which on an
# AVX-512 build host bakes EVEX into an artifact bound for machines that cannot
# execute it (5092 EVEX instructions were measured).
#
# The earlier attempt set SOURCE_DATE_EPOCH, which makes GGML_NATIVE_DEFAULT=OFF,
# and that lands on the INS_ENB OFF branch too. The CPU backend was then built
# with NO SIMD AT ALL. Measured on an Intel Core Ultra 5 226V with
# large-v3-turbo Q5_0 and 8 threads: 3630 ms of audio took 128235 ms to
# transcribe, real-time factor 0.028, and the machine stalled under full load.
#
# A CMake toolchain file forces GGML_NATIVE=OFF in the cache while leaving
# GGML_NATIVE_DEFAULT alone, so INS_ENB stays ON and the backend is compiled
# with SSE4.2, F16C, FMA, BMI2, AVX and AVX2, and nothing above AVX2. CMake
# reads CMAKE_TOOLCHAIN_FILE from the environment and runs it before the
# project, early enough that option() will not override it. No patched crate.
$toolchain = Join-Path $PSScriptRoot "..\..\..\.github\cmake\portable-x86.cmake"
if (-not (Test-Path -LiteralPath $toolchain)) {
    throw "toolchain file not found: $toolchain"
}
$env:CMAKE_TOOLCHAIN_FILE = (Resolve-Path -LiteralPath $toolchain).Path
Write-Host "    CMAKE_TOOLCHAIN_FILE=$($env:CMAKE_TOOLCHAIN_FILE)" -ForegroundColor DarkGray
Write-Host "    (ggml GGML_NATIVE=OFF, AVX2 on, no AVX-512, no -march=native)" -ForegroundColor DarkGray

# Guard against the regression that caused the stall: if SOURCE_DATE_EPOCH is
# set anywhere, GGML_NATIVE_DEFAULT flips to OFF and INS_ENB goes back to OFF.
if ($env:SOURCE_DATE_EPOCH) {
    Write-Host "    clearing SOURCE_DATE_EPOCH ($($env:SOURCE_DATE_EPOCH)): it would disable ggml's AVX2 kernels" -ForegroundColor Yellow
    Remove-Item Env:\SOURCE_DATE_EPOCH -ErrorAction SilentlyContinue
}

# Defence in depth only: rustc's baseline x86-64 is sse2 and never emitted
# AVX-512, so this is not what caused the crash. Kept so a future toolchain
# default cannot reintroduce it.
# Applies unless the caller already set RUSTFLAGS deliberately.
$baselineFlags = @(
    "-C", "target-feature=-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl",
    "-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512bitalg,-avx512vpopcntdq"
) -join " "

if ($env:RUSTFLAGS) {
    Write-Host "    RUSTFLAGS already set, leaving it alone: $env:RUSTFLAGS" -ForegroundColor DarkGray
} else {
    $env:RUSTFLAGS = $baselineFlags
    Write-Host "    Baseline CPU target: $baselineFlags" -ForegroundColor DarkGray
}

$tauriArgs = @()

if ($Debug) {
    $tauriArgs += "--debug"
}

# Cargo takes one --features list, so combined switches are joined with a comma.
$features = @()

if ($Cuda)   { $features += "cuda" }
if ($Vulkan) { $features += "vulkan" }

if ($features.Count -gt 0) {
    $tauriArgs += "--features"
    $tauriArgs += ($features -join ",")
}

Write-Host "    Running: npm run tauri build $($tauriArgs -join ' ')" -ForegroundColor DarkGray

if ($tauriArgs.Count -gt 0) {
    npm run tauri build -- @tauriArgs
} else {
    npm run tauri build
}

if ($LASTEXITCODE -ne 0) { Write-Fail "Build failed" }

# ── Report ────────────────────────────────────────────────────────────────────

Write-Step "Build complete"

$bundleDir = Join-Path $PSScriptRoot "..\src-tauri\target\release\bundle"
if ($Debug) {
    $bundleDir = Join-Path $PSScriptRoot "..\src-tauri\target\debug\bundle"
}

if (Test-Path $bundleDir) {
    $artifacts = Get-ChildItem $bundleDir -Recurse -Include "*.exe","*.msi" |
        Where-Object { $_.Name -notlike "*-setup-stub*" }
    foreach ($a in $artifacts) {
        $size = "{0:N1} MB" -f ($a.Length / 1MB)
        Write-Host "    $($a.FullName) ($size)" -ForegroundColor Green
    }
} else {
    Write-Host "    Bundle directory not found at $bundleDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan

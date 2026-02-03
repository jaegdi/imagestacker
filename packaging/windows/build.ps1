# ImageStacker Windows build script
# Creates a portable ZIP containing imagestacker.exe and docs.
#
# Usage (PowerShell):
#   ./packaging/windows/build.ps1
#
# Notes:
# - Requires Rust toolchain installed (rustup).
# - OpenCV is a native dependency; the produced exe will still need OpenCV DLLs
#   depending on how your opencv-rust is configured on Windows.

$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '../..')
$distDir = Join-Path $projectRoot 'dist/windows'
$binDir = Join-Path $distDir 'bin'
$docDir = Join-Path $distDir 'doc'

Write-Host "Building ImageStacker (Windows release)..." -ForegroundColor Cyan
Push-Location $projectRoot
cargo build --release
Pop-Location

New-Item -ItemType Directory -Force -Path $binDir | Out-Null
New-Item -ItemType Directory -Force -Path $docDir | Out-Null

Copy-Item -Force (Join-Path $projectRoot 'target/release/imagestacker.exe') $binDir

# Docs
if (Test-Path (Join-Path $projectRoot 'USER_MANUAL.md')) {
  Copy-Item -Force (Join-Path $projectRoot 'USER_MANUAL.md') $docDir
}
if (Test-Path (Join-Path $projectRoot 'README.md')) {
  Copy-Item -Force (Join-Path $projectRoot 'README.md') $docDir
}

# Optional icon assets for bundling
if (Test-Path (Join-Path $projectRoot 'icons/imagestacker_icon.png')) {
  Copy-Item -Force (Join-Path $projectRoot 'icons/imagestacker_icon.png') $distDir
}

# Create ZIP
$zipPath = Join-Path $projectRoot 'dist/imagestacker-windows-x86_64.zip'
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($distDir, $zipPath)

Write-Host "Created: $zipPath" -ForegroundColor Green

# Dot-source from your application's directory. Does not write to the Go module
# cache, change the current directory, or install persistent environment values.
param(
    [string]$DependencyRoot = (Split-Path $PSScriptRoot -Parent),
    [string]$CompilerBin
)
$ErrorActionPreference = 'Stop'
$inpaintDeps = (Resolve-Path -LiteralPath $DependencyRoot).Path
$inpaintNative = Join-Path $inpaintDeps 'native\windows-amd64'
$inpaintInclude = (Join-Path $inpaintNative 'include\opencv4').Replace('\','/')
$inpaintLib = (Join-Path $inpaintNative 'lib').Replace('\','/')
$inpaintBin = Join-Path $inpaintNative 'bin'
foreach ($required in @(
    (Join-Path $inpaintInclude 'opencv2\core.hpp'),
    (Join-Path $inpaintLib 'libopencv_core4120.dll.a'),
    (Join-Path $inpaintLib 'libopencv_imgproc4120.dll.a'),
    (Join-Path $inpaintLib 'libopencv_photo4120.dll.a'),
    (Join-Path $inpaintBin 'onnxruntime.dll'),
    (Join-Path $inpaintDeps 'models\ch_PP-OCRv4_det_infer.onnx')
)) { if (!(Test-Path -LiteralPath $required)) { throw "Missing dependency: $required. Run setup-windows.ps1 in a writable dependency checkout first." } }
if ($CompilerBin) { $env:PATH = $CompilerBin + ';' + $env:PATH }
foreach ($command in @('gcc','g++')) { if (!(Get-Command $command -ErrorAction SilentlyContinue)) { throw "$command is required; pass -CompilerBin" } }
$env:PATH = $inpaintBin + ';' + $env:PATH
$env:CGO_ENABLED = '1'
$env:CC = 'gcc'
$env:CXX = 'g++'
# Quotes are interpreted by cgo, so dependency directories may contain spaces.
$env:CGO_CXXFLAGS = ('"-I' + $inpaintInclude + '" ' + $env:CGO_CXXFLAGS).Trim()
$env:CGO_LDFLAGS = ('"-L' + $inpaintLib + '" ' + $env:CGO_LDFLAGS).Trim()
$env:INPAINT_MODEL = Join-Path $inpaintDeps 'models\ch_PP-OCRv4_det_infer.onnx'
$env:INPAINT_RUNTIME = Join-Path $inpaintBin 'onnxruntime.dll'
Write-Host 'Go module native dependencies are configured for this PowerShell session.'

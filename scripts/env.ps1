param([string]$CompilerBin)
$textInpaintRoot = Split-Path $PSScriptRoot -Parent
$nativeBin = Join-Path $textInpaintRoot 'native\windows-amd64\bin'
$env:PATH = $nativeBin + ';' + $env:PATH
if ($CompilerBin) { $env:PATH = $CompilerBin + ';' + $env:PATH }
$env:CGO_ENABLED = '1'
$env:CC = 'gcc'
$env:CXX = 'g++'
$env:GOCACHE = Join-Path $textInpaintRoot 'build\go-cache'
$env:GOMODCACHE = Join-Path $textInpaintRoot 'build\go-mod'
$env:GOTMPDIR = Join-Path $textInpaintRoot 'build\go-tmp'
New-Item -ItemType Directory -Path $env:GOCACHE,$env:GOMODCACHE,$env:GOTMPDIR -Force | Out-Null

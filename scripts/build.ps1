param([string]$CompilerBin)
$ErrorActionPreference = 'Stop'
$textInpaintRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'env.ps1') -CompilerBin $CompilerBin
Push-Location $textInpaintRoot
try {
    New-Item -ItemType Directory -Path 'bin' -Force | Out-Null
    & go build -o 'bin\textinpaint.exe' './cmd/textinpaint'
    if ($LASTEXITCODE -ne 0) { throw 'Go build failed' }
    & go build -o 'bin\textinpaint-server.exe' './cmd/textinpaint-server'
    if ($LASTEXITCODE -ne 0) { throw 'Go server build failed' }
    Get-ChildItem -LiteralPath 'native\windows-amd64\bin' -Filter '*.dll' | Copy-Item -Destination 'bin' -Force
} finally { Pop-Location }

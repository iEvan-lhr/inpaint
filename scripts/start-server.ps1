param([string]$ListenAddress = '0.0.0.0:8080')
$ErrorActionPreference = 'Stop'
$libraryRoot = Split-Path $PSScriptRoot -Parent
$server = Join-Path $libraryRoot 'bin\textinpaint-server.exe'
if (!(Test-Path -LiteralPath $server)) { throw 'Server executable is missing. Run scripts/build.ps1 first.' }
& $server -listen $ListenAddress
if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }

[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' })
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run-optimization-sweep.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $runner `
    -PythonExe $PythonExe `
    -SumoHome $SumoHome `
    -Config 'configs\hybrid-fusion-sweep.toml' `
    -ResultRoot 'results\verified\hybrid-fusion-validation' `
    -ExpectedRuns 5
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid fusion sweep exited with code $LASTEXITCODE"
}

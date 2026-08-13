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
    -Config 'configs\decision-ablation-sweep.toml' `
    -ResultRoot 'results\verified\decision-ablation-sweep' `
    -ExpectedRuns 16
if ($LASTEXITCODE -ne 0) {
    throw "Decision ablation exited with code $LASTEXITCODE"
}

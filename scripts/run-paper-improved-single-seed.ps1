[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$SampleIntervalSeconds = 5
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run-paper-single-seed.ps1'
& $runner `
    -PythonExe $PythonExe `
    -SumoHome $SumoHome `
    -Config 'configs\paper-improved-single-seed.toml' `
    -ResultSubdir 'paper-improved-fusion-single-seed' `
    -ExpectedRuns 15 `
    -SampleIntervalSeconds $SampleIntervalSeconds

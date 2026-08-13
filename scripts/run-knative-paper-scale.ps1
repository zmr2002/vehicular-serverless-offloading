[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Profile = 'knative',
    [int[]]$VehicleCounts,
    [string]$ResumeSession,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run-knative-validation.ps1'
$parameters = @{
    PythonExe = $PythonExe
    SumoHome = $SumoHome
    Profile = $Profile
    Config = 'configs\knative-paper-scale.toml'
}
if ($VehicleCounts.Count -gt 0) {
    $parameters.VehicleCounts = $VehicleCounts
}
if ($PreflightOnly) {
    $parameters.PreflightOnly = $true
}
if ($ResumeSession) {
    $parameters.ResumeSession = $ResumeSession
}

& $runner @parameters
if ($LASTEXITCODE -ne 0) {
    throw "Paper-scale Knative validation exited with code $LASTEXITCODE."
}

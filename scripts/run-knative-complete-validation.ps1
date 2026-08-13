[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Profile = 'knative',
    [int[]]$VehicleCounts,
    [string]$ResumeValidationSession,
    [switch]$PreflightOnly,
    [switch]$SkipDeployment,
    [switch]$SkipDockerSmoke,
    [switch]$SkipBenchmark,
    [switch]$SkipUnitTests
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $repo.Replace('\', '/')
$commit = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
    throw 'Unable to resolve the current Git commit.'
}
$status = @(& git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect repository status.'
}
if ($status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before running the resumable pipeline.'
}

$session = Join-Path $repo (
    'results\verified\knative-complete-validation\run-' + $commit.Substring(0, 8)
)
$markers = Join-Path $session 'completed-stages'
New-Item -ItemType Directory -Force -Path $markers | Out-Null
$transcript = Join-Path $session 'pipeline.log'

if (-not ('NativeExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class NativeExecutionState
{
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

function Invoke-ValidationStage {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [scriptblock]$Action
    )

    $marker = Join-Path $markers "$Name.json"
    if (Test-Path -LiteralPath $marker) {
        Write-Host "SKIP $Name"
        return
    }

    Write-Host "START $Name"
    $started = Get-Date
    try {
        & $Action
        @{
            stage = $Name
            started_at = $started.ToUniversalTime().ToString('o')
            completed_at = (Get-Date).ToUniversalTime().ToString('o')
            git_commit = $commit
        } |
            ConvertTo-Json |
            Set-Content -LiteralPath $marker -Encoding utf8
        Write-Host "DONE $Name"
    } catch {
        @{
            stage = $Name
            started_at = $started.ToUniversalTime().ToString('o')
            failed_at = (Get-Date).ToUniversalTime().ToString('o')
            git_commit = $commit
            error = $_.Exception.Message
        } |
            ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $session 'last-failure.json') -Encoding utf8
        throw
    }
}

$paperParameters = @{
    PythonExe = $python
    SumoHome = $SumoHome
    Profile = $Profile
}
if ($VehicleCounts.Count -gt 0) {
    $paperParameters.VehicleCounts = $VehicleCounts
}
if ($ResumeValidationSession) {
    $paperParameters.ResumeSession = $ResumeValidationSession
}

Write-Host "SESSION $session"
Write-Host 'Stages are resumable for this Git commit.'
[void][NativeExecutionState]::SetThreadExecutionState([uint32]2147483649)
Write-Host 'Automatic system sleep is disabled until this pipeline exits; the display may turn off.'
Start-Transcript -LiteralPath $transcript -Append | Out-Null
try {
    if ($PreflightOnly) {
        $paperParameters.PreflightOnly = $true
        & (Join-Path $PSScriptRoot 'run-knative-paper-scale.ps1') @paperParameters
        Write-Host 'PREFLIGHT PIPELINE COMPLETE'
        return
    }

    if (-not $SkipUnitTests) {
        Invoke-ValidationStage '01-unit-tests' {
            Push-Location $repo
            try {
                & $python -m unittest discover -s tests
                if ($LASTEXITCODE -ne 0) {
                    throw "Test suite exited with code $LASTEXITCODE."
                }
            } finally {
                Pop-Location
            }
        }
    }

    if (-not $SkipDeployment) {
        Invoke-ValidationStage '02-function-deployment' {
            & (Join-Path $PSScriptRoot 'deploy-knative.ps1') -Profile $Profile
        }
    }

    Write-Host 'START 03-paper-scale-preflight'
    $preflight = @{} + $paperParameters
    $preflight.PreflightOnly = $true
    & (Join-Path $PSScriptRoot 'run-knative-paper-scale.ps1') @preflight
    Write-Host 'DONE 03-paper-scale-preflight'

    if (-not $SkipDockerSmoke) {
        Invoke-ValidationStage '04-docker-simulator-smoke' {
            & (Join-Path $PSScriptRoot 'run-knative-smoke.ps1') -Profile $Profile
        }
    }

    if (-not $SkipBenchmark) {
        Invoke-ValidationStage '05-cold-warm-burst-benchmark' {
            & (Join-Path $PSScriptRoot 'benchmark-knative.ps1') `
                -Profile $Profile `
                -PythonExe $python `
                -RequestsPerLevel 100 `
                -WorkUnits 250000
        }
    }

    Invoke-ValidationStage '06-paper-scale-paired-validation' {
        & (Join-Path $PSScriptRoot 'run-knative-paper-scale.ps1') @paperParameters
    }

    @{
        status = 'complete'
        completed_at = (Get-Date).ToUniversalTime().ToString('o')
        git_commit = $commit
        vehicle_counts = if ($VehicleCounts.Count -gt 0) {
            @($VehicleCounts)
        } else {
            @(1000, 2000, 4000)
        }
        stages = @(
            'unit tests'
            'function deployment'
            'paper-scale preflight'
            'Docker simulator smoke'
            'cold/warm and concurrency 1/10/50 benchmark'
            'analytical/live paper-scale paired validation'
        )
    } |
        ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $session 'pipeline-summary.json') -Encoding utf8
    Write-Host "COMPLETE $session"
} finally {
    Stop-Transcript | Out-Null
    [void][NativeExecutionState]::SetThreadExecutionState([uint32]2147483648)
}

[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Profile = 'knative',
    [int[]]$Replicates = @(1, 2, 3),
    [int]$AnalyticalParallelism = 6,
    [string]$ResumeValidationSession,
    [switch]$AllSixReplicates,
    [switch]$PreflightOnly,
    [switch]$SkipDeployment,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$selectedReplicates = if ($AllSixReplicates) {
    @(1, 2, 3, 4, 5, 6)
} else {
    @($Replicates)
}
if ($selectedReplicates.Count -eq 0) {
    throw 'At least one final-experiment replicate must be selected.'
}
if ($selectedReplicates.Count -ne @($selectedReplicates | Sort-Object -Unique).Count) {
    throw 'Replicate numbers must be unique.'
}
if (@($selectedReplicates | Where-Object { $_ -lt 1 -or $_ -gt 6 }).Count -gt 0) {
    throw 'Replicate numbers must be within 1..6.'
}
if ($AnalyticalParallelism -lt 1 -or $AnalyticalParallelism -gt 6) {
    throw 'AnalyticalParallelism must be within 1..6 on the reviewed host.'
}

$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $repo.Replace('\', '/')

$status = @(& git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect repository status.'
}
if ($status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before resumable validation.'
}

$generatedConfig = Join-Path $repo (
    'configs\.knative-final-decoupled-v2.generated.toml'
)
$validationOutput = Join-Path $repo (
    'results\verified\serverless-final-decoupled-v2-three-seed'
)
if ($AllSixReplicates) {
    $validationOutput = Join-Path $repo (
        'results\verified\serverless-final-decoupled-v2-six-seed'
    )
}

if (-not ('FinalKnativeExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class FinalKnativeExecutionState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

[void][FinalKnativeExecutionState]::SetThreadExecutionState([uint32]2147483649)
try {
    if (-not $SkipTests) {
        Push-Location $repo
        try {
            & $python -m unittest `
                tests.test_prepare_final_serverless_config `
                tests.test_knative_validation_runner
            if ($LASTEXITCODE -ne 0) {
                throw "Targeted tests exited with code $LASTEXITCODE."
            }
        } finally {
            Pop-Location
        }
    }

    $prepareArguments = @(
        (Join-Path $PSScriptRoot 'prepare-final-serverless-config.py'),
        '--final-config',
        (Join-Path $repo 'configs\final-decoupled-v2.toml'),
        '--base-config',
        (Join-Path $repo 'configs\hybrid-decoupled.toml'),
        '--output',
        $generatedConfig,
        '--validation-output-dir',
        $validationOutput,
        '--analytical-parallelism',
        $AnalyticalParallelism.ToString(),
        '--replicates'
    )
    $prepareArguments += @(
        $selectedReplicates | ForEach-Object { $_.ToString() }
    )
    & $python @prepareArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not generate the final-model Knative configuration.'
    }

    if (-not $SkipDeployment -and -not $PreflightOnly) {
        & (Join-Path $PSScriptRoot 'deploy-knative.ps1') -Profile $Profile
        if ($LASTEXITCODE -ne 0) {
            throw 'Knative deployment failed.'
        }
    }

    $validationParameters = @{
        PythonExe = $python
        SumoHome = $sumo
        Profile = $Profile
        Config = $generatedConfig
    }
    if ($ResumeValidationSession) {
        $validationParameters.ResumeSession = $ResumeValidationSession
    }
    if ($PreflightOnly) {
        $validationParameters.PreflightOnly = $true
    }

    Write-Host (
        'Final model: adequacy-arbitrated Hybrid over the paired pure-DQN ' +
        'checkpoint.'
    )
    Write-Host (
        'Replicates: ' + ($selectedReplicates -join ', ') +
        '; vehicles: 1000, 2000, 4000; modes: analytical, replay, closed loop.'
    )
    & (Join-Path $PSScriptRoot 'run-knative-validation.ps1') `
        @validationParameters
    if ($LASTEXITCODE -ne 0) {
        throw 'Final-model Knative validation failed.'
    }
} finally {
    [void][FinalKnativeExecutionState]::SetThreadExecutionState(
        [uint32]2147483648
    )
}

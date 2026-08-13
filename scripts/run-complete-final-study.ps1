[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Profile = 'knative',
    [int]$Parallelism = 6,
    [switch]$PreflightOnly,
    [switch]$SkipDeployment,
    [switch]$SkipOldModelServerless,
    [switch]$SkipAblation,
    [switch]$SkipFinalExperiment,
    [switch]$SkipFinalServerless,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
if ($Parallelism -ne 6) {
    Write-Host "Parallelism was set to $Parallelism; the reviewed default is 6."
}

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
    throw 'Commit or remove working-tree changes before the resumable final study.'
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
$freeGiB = $drive.AvailableFreeSpace / 1GB
$reserveGiB = 12.0
$estimatedGiB = 4.0
if (-not $PreflightOnly -and $freeGiB -lt ($reserveGiB + $estimatedGiB)) {
    throw (
        'Only {0:N2} GiB is free; the final study requires {1:N2} GiB ' +
        'including the protected reserve.' -f
        $freeGiB, ($reserveGiB + $estimatedGiB)
    )
}
$diskMessage = (
    'Disk free before run: {0:N2} GiB. Estimated retained output is below ' +
    '{1:N2} GiB; protected reserve is {2:N2} GiB.'
)
Write-Host ($diskMessage -f $freeGiB, $estimatedGiB, $reserveGiB)
Write-Host (
    'Analytical work uses a six-process global budget. Knative replay and ' +
    'closed-loop validation remain sequential so platform latency is measurable.'
)

$session = Join-Path $repo (
    'results\verified\complete-final-study\run-' + $commit.Substring(0, 8)
)
$markers = Join-Path $session 'completed-stages'
New-Item -ItemType Directory -Force -Path $markers | Out-Null
$transcript = Join-Path $session 'pipeline.log'

if (-not ('FinalStudyExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class FinalStudyExecutionState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

function Invoke-StudyStage {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Action
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
        } | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
        Write-Host "DONE $Name"
    } catch {
        @{
            stage = $Name
            failed_at = (Get-Date).ToUniversalTime().ToString('o')
            error = $_.Exception.Message
            git_commit = $commit
        } | ConvertTo-Json | Set-Content `
            -LiteralPath (Join-Path $session 'last-failure.json') `
            -Encoding utf8
        throw
    }
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command exited with code $LASTEXITCODE."
    }
}

$ablationConfig = Join-Path $repo 'configs\follower-state-ablation.toml'
$finalConfig = Join-Path $repo 'configs\final-three-seed.toml'
$selectedPath = Join-Path $repo (
    'results\verified\follower-state-ablation\selected-profile.json'
)
$generatedKnative = Join-Path $repo (
    'configs\.knative-final-three-seed.generated.toml'
)

[void][FinalStudyExecutionState]::SetThreadExecutionState([uint32]2147483649)
Start-Transcript -LiteralPath $transcript -Append | Out-Null
try {
    Invoke-StudyStage '01-tests-and-dry-runs' {
        Push-Location $repo
        try {
            Invoke-Python -Arguments @('-m', 'unittest', 'discover', '-s', 'tests')
            Invoke-Python -Arguments @(
                (Join-Path $repo 'scripts\run-follower-state-ablation.py'),
                '--config', $ablationConfig,
                '--parallelism', $Parallelism,
                '--dry-run'
            )
            Invoke-Python -Arguments @(
                (Join-Path $repo 'scripts\run-final-multiseed.py'),
                '--config', $finalConfig,
                '--parallelism', $Parallelism,
                '--dry-run'
            )
        } finally {
            Pop-Location
        }
    }

    if ($PreflightOnly) {
        Write-Host 'PREFLIGHT COMPLETE. No long experiment or deployment was run.'
        return
    }

    Invoke-StudyStage '02-pricing-batch-performance' {
        Invoke-Python -Arguments @(
            (Join-Path $repo 'scripts\benchmark-pricing-batch.py'),
            '--config',
            (Join-Path $repo 'configs\pricing-batch-performance.toml')
        )
    }

    if (-not $SkipDeployment) {
        Invoke-StudyStage '03-knative-deployment' {
            & (Join-Path $PSScriptRoot 'deploy-knative.ps1') -Profile $Profile
            if ($LASTEXITCODE -ne 0) { throw 'Knative deployment failed.' }
        }
    }

    if (-not $SkipBenchmark) {
        Invoke-StudyStage '04-cold-warm-burst-benchmark' {
            & (Join-Path $PSScriptRoot 'benchmark-knative.ps1') `
                -Profile $Profile `
                -PythonExe $python `
                -RequestsPerLevel 100 `
                -WorkUnits 250000
            if ($LASTEXITCODE -ne 0) { throw 'Knative benchmark failed.' }
        }
    }

    if (-not $SkipOldModelServerless) {
        Invoke-StudyStage '05-pre-serverless-model-three-seed' {
            & (Join-Path $PSScriptRoot 'run-knative-validation.ps1') `
                -PythonExe $python `
                -SumoHome $sumo `
                -Profile $Profile `
                -Config 'configs\knative-pre-serverless-three-seed.toml'
            if ($LASTEXITCODE -ne 0) { throw 'Old-model Serverless validation failed.' }
        }
    }

    if (-not $SkipAblation) {
        Invoke-StudyStage '06-follower-state-ablation' {
            Invoke-Python -Arguments @(
                (Join-Path $repo 'scripts\run-follower-state-ablation.py'),
                '--config', $ablationConfig,
                '--parallelism', $Parallelism
            )
        }
    }
    if (-not (Test-Path -LiteralPath $selectedPath)) {
        throw "Selected profile is missing: $selectedPath"
    }
    $selection = Get-Content -Raw -LiteralPath $selectedPath | ConvertFrom-Json
    $selectedConfig = Join-Path $repo ('configs\' + $selection.config)
    if (-not (Test-Path -LiteralPath $selectedConfig)) {
        throw "Selected base config is missing: $selectedConfig"
    }
    Write-Host "Selected final model profile: $($selection.name)"

    if (-not $SkipFinalExperiment) {
        Invoke-StudyStage '07-final-analytical-three-seed' {
            Invoke-Python -Arguments @(
                (Join-Path $repo 'scripts\run-final-multiseed.py'),
                '--config', $finalConfig,
                '--base-config', $selectedConfig,
                '--parallelism', $Parallelism
            )
        }
    }

    if (-not $SkipFinalServerless) {
        Invoke-StudyStage '08-generate-final-serverless-config' {
            Invoke-Python -Arguments @(
                (Join-Path $repo 'scripts\prepare-final-serverless-config.py'),
                '--final-config', $finalConfig,
                '--base-config', $selectedConfig,
                '--output', $generatedKnative
            )
        }
        Invoke-StudyStage '09-final-model-serverless-three-seed' {
            & (Join-Path $PSScriptRoot 'run-knative-validation.ps1') `
                -PythonExe $python `
                -SumoHome $sumo `
                -Profile $Profile `
                -Config $generatedKnative
            if ($LASTEXITCODE -ne 0) { throw 'Final Serverless validation failed.' }
        }
    }

    @{
        status = 'complete'
        completed_at = (Get-Date).ToUniversalTime().ToString('o')
        git_commit = $commit
        analytical_parallelism = $Parallelism
        selected_profile = $selection.name
        selected_config = $selection.config
        serverless_modes = @('analytical', 'knative_replay', 'knative_closed_loop')
        seeds_per_experiment = 3
    } | ConvertTo-Json | Set-Content `
        -LiteralPath (Join-Path $session 'pipeline-summary.json') `
        -Encoding utf8
    Write-Host "COMPLETE $session"
} finally {
    Stop-Transcript | Out-Null
    [void][FinalStudyExecutionState]::SetThreadExecutionState([uint32]2147483648)
}

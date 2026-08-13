[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$Parallelism = 6,
    [switch]$SkipScreen,
    [switch]$DryRun
)

# Plan B in one resumable command:
#   1. Epsilon A/B screen: retrain the pure DQN at 2,000 vehicles on the
#      SAME training seeds as the fresh matrix (31641-43) with the per-step
#      exploration schedule, evaluate on the paired seeds (84-86), and select
#      the training recipe by the pre-declared +0.25 pp mean-success rule.
#   2. Six-replicate final paired matrix (configs/final-decoupled-v2.toml,
#      seeds 31651-56 / 87-92) with the winning recipe: DQN training only,
#      hybrid evaluates the adequacy arbitration over each replicate's
#      checkpoint. All replicates are reported; no checkpoint selection.
# Re-running the same command resumes completed cases.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path

$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $repo.Replace('\', '/')

$status = @(& git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect the repository status.'
}
if (-not $DryRun -and $status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before the final experiment.'
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
$freeGiB = $drive.AvailableFreeSpace / 1GB
if (-not $DryRun -and $freeGiB -lt 20.0) {
    throw ("Only {0:N2} GiB free; at least 20 GiB is required." -f $freeGiB)
}
Write-Host ("Disk free before run: {0:N2} GiB." -f $freeGiB)

if (-not ('FinalV2ExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class FinalV2ExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

try {
    if (-not $DryRun) {
        [void][FinalV2ExecutionState]::SetThreadExecutionState([uint32]2147483649)
    }

    $baseConfig = Join-Path $repo 'configs\hybrid-decoupled.toml'
    if (-not $SkipScreen) {
        Write-Host '==== Epsilon training-recipe screen ===='
        $screenArguments = @((Join-Path $repo 'scripts\run_eps_screen.py'))
        if ($DryRun) { $screenArguments += '--dry-run' }
        $screen = & $python @screenArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Epsilon screen exited with code $LASTEXITCODE."
        }
        $screen | ForEach-Object { Write-Host $_ }
        $baseConfig = ($screen | Select-Object -Last 1).Trim()
    }
    Write-Host "Selected training recipe: $baseConfig"

    Write-Host '==== Six-replicate final matrix ===='
    $finalArguments = @(
        (Join-Path $repo 'scripts\run-final-multiseed.py'),
        '--config', (Join-Path $repo 'configs\final-decoupled-v2.toml'),
        '--base-config', $baseConfig,
        '--parallelism', $Parallelism
    )
    if ($DryRun) { $finalArguments += '--dry-run' }
    & $python @finalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Final matrix exited with code $LASTEXITCODE."
    }
}
finally {
    if (-not $DryRun) {
        [void][FinalV2ExecutionState]::SetThreadExecutionState([uint32]2147483648)
    }
}

Write-Host 'Plan B finished.'
Write-Host ('Final: ' + (Join-Path $repo 'results\verified\final-decoupled-v2\final-summary.md'))

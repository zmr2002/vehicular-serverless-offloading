[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$Parallelism = 6,
    [int]$ArmWorkers = 3,
    [switch]$SkipArms,
    [switch]$SkipFinal,
    [switch]$DryRun
)

# Complete decoupled-Hybrid validation in one resumable command:
#   1. Cross-checkpoint arms on the existing final-matrix checkpoints
#      (no training): dqnckpt, dqnckpt-reliability, internal-reliability
#      at 1000/2000/4000 vehicles for replicates 1-3.
#   2. Fresh-seed final paired multi-seed matrix (configs/final-decoupled.toml):
#      trains only the pure DQN; hybrid_stackelberg evaluates the arbitration
#      over that checkpoint. Seeds 31641-43 / 84-86 are unused by the
#      diagnosis and carry the confirmatory claim.
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
    throw 'Commit or remove working-tree changes before the validation run.'
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
$freeGiB = $drive.AvailableFreeSpace / 1GB
if (-not $DryRun -and $freeGiB -lt 17.0) {
    throw ("Only {0:N2} GiB free; at least 17 GiB is required." -f $freeGiB)
}
Write-Host ("Disk free before run: {0:N2} GiB." -f $freeGiB)

if (-not ('DecoupledValidationExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class DecoupledValidationExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "==== $Label ===="
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label exited with code $LASTEXITCODE."
    }
}

try {
    if (-not $DryRun) {
        [void][DecoupledValidationExecutionState]::SetThreadExecutionState(
            [uint32]2147483649
        )
    }

    if (-not $SkipArms) {
        foreach ($arm in @('dqnckpt', 'dqnckpt-reliability', 'internal-reliability')) {
            $armArguments = @(
                (Join-Path $repo 'scripts\run_cross_checkpoint_eval.py'),
                '--arm', $arm,
                '--workers', $ArmWorkers
            )
            if ($DryRun) { $armArguments += '--dry-run' }
            Invoke-Step "Cross-checkpoint arm $arm" $armArguments
        }
        if (-not $DryRun) {
            Invoke-Step 'Arm summary' @(
                (Join-Path $repo 'scripts\summarize_cross_checkpoint_arms.py')
            )
        }
    }

    if (-not $SkipFinal) {
        $finalArguments = @(
            (Join-Path $repo 'scripts\run-final-multiseed.py'),
            '--config', (Join-Path $repo 'configs\final-decoupled.toml'),
            '--parallelism', $Parallelism
        )
        if ($DryRun) { $finalArguments += '--dry-run' }
        Invoke-Step 'Fresh-seed decoupled final matrix' $finalArguments
    }
}
finally {
    if (-not $DryRun) {
        [void][DecoupledValidationExecutionState]::SetThreadExecutionState(
            [uint32]2147483648
        )
    }
}

Write-Host 'Validation pipeline finished.'
Write-Host ('Arms:  ' + (Join-Path $repo 'results\verified\hybrid-cross-checkpoint-eval\arms-summary.md'))
Write-Host ('Final: ' + (Join-Path $repo 'results\verified\final-decoupled\final-summary.md'))

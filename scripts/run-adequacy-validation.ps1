[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$ArmWorkers = 3,
    [switch]$SkipScreening,
    [switch]$SkipFresh,
    [switch]$DryRun
)

# Adequacy-arbitration validation in one resumable command:
#   1. Screening arms on the study checkpoints (evaluation seeds 81-83, no
#      training): dqnckpt-adequacy (adaptive A^p damping), dqnckpt-cap
#      (structural bounded game evidence), and dqnckpt-damping (damping
#      without the refutation defense; ablation only).
#   2. Pre-declared selection: the candidate with the best WORST-scale mean
#      margin over the strongest baseline (ties by overall mean margin).
#   3. Fresh re-evaluation: hybrid_stackelberg with the winning arbitration
#      on the final-decoupled pure-DQN checkpoints (evaluation seeds 84-86),
#      merged with the untouched baseline rows into
#      results\verified\final-decoupled\corrected-final-summary.md.
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
if (-not $DryRun -and $freeGiB -lt 15.0) {
    throw ("Only {0:N2} GiB free; at least 15 GiB is required." -f $freeGiB)
}
Write-Host ("Disk free before run: {0:N2} GiB." -f $freeGiB)

if (-not ('AdequacyValidationExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AdequacyValidationExecutionState {
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
        [void][AdequacyValidationExecutionState]::SetThreadExecutionState(
            [uint32]2147483649
        )
    }

    if (-not $SkipScreening) {
        foreach ($arm in @('dqnckpt-adequacy', 'dqnckpt-cap', 'dqnckpt-damping')) {
            $armArguments = @(
                (Join-Path $repo 'scripts\run_cross_checkpoint_eval.py'),
                '--arm', $arm,
                '--source', 'study',
                '--workers', $ArmWorkers
            )
            if ($DryRun) { $armArguments += '--dry-run' }
            Invoke-Step "Screening arm $arm" $armArguments
        }
        if (-not $DryRun) {
            Invoke-Step 'Arm summary' @(
                (Join-Path $repo 'scripts\summarize_cross_checkpoint_arms.py')
            )
        }
    }

    if (-not $SkipFresh) {
        $freshArm = 'dqnckpt-adequacy'
        if (-not $DryRun) {
            Write-Host '==== Pre-declared candidate selection ===='
            $selection = & $python (Join-Path $repo 'scripts\select_screening_winner.py')
            if ($LASTEXITCODE -ne 0) {
                throw "Candidate selection exited with code $LASTEXITCODE."
            }
            $selection | ForEach-Object { Write-Host $_ }
            $freshArm = ($selection | Select-Object -Last 1).Trim()
        }
        $freshArguments = @(
            (Join-Path $repo 'scripts\run_cross_checkpoint_eval.py'),
            '--arm', $freshArm,
            '--source', 'fresh',
            '--workers', $ArmWorkers
        )
        if ($DryRun) { $freshArguments += '--dry-run' }
        Invoke-Step "Fresh-seed hybrid re-evaluation ($freshArm)" $freshArguments
        if (-not $DryRun) {
            Invoke-Step 'Corrected final summary' @(
                (Join-Path $repo 'scripts\summarize_fresh_final.py')
            )
        }
    }
}
finally {
    if (-not $DryRun) {
        [void][AdequacyValidationExecutionState]::SetThreadExecutionState(
            [uint32]2147483648
        )
    }
}

Write-Host 'Adequacy validation finished.'
Write-Host ('Arms:      ' + (Join-Path $repo 'results\verified\hybrid-cross-checkpoint-eval\arms-summary.md'))
Write-Host ('Corrected: ' + (Join-Path $repo 'results\verified\final-decoupled\corrected-final-summary.md'))

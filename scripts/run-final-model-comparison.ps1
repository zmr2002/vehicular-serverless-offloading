[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$Parallelism = 4,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$runner = Join-Path $repo 'scripts\run-training-evaluation.py'
$output = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\final-model-comparison')
)
$verified = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified')
)
if (-not $output.StartsWith(
    $verified + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Unexpected result path: $output"
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
if (-not $DryRun -and $status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before running the resumable comparison.'
}

if (-not $DryRun) {
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
    $freeGiB = $drive.AvailableFreeSpace / 1GB
    if ($freeGiB -lt 16.0) {
        $diskError = "Only {0:N2} GiB is free on {1}; at least 16 GiB is required."
        throw ($diskError -f $freeGiB, $drive.Name)
    }
    $storageMessage = (
        "Disk free before run: {0:N2} GiB. Training task logs are disabled; " +
        "evaluation task logs use a 0.1% sample."
    )
    Write-Host ($storageMessage -f $freeGiB)
}

$configs = @(
    'configs\final-model-comparison-baselines.toml',
    'configs\final-model-comparison-thesis-hybrid.toml',
    'configs\final-model-comparison-enhanced-hybrid.toml'
)

if (-not ('ModelComparisonExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class ModelComparisonExecutionState
{
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

if (-not $DryRun) {
    [void][ModelComparisonExecutionState]::SetThreadExecutionState(
        [uint32]2147483649
    )
}
try {
    for ($index = 0; $index -lt $configs.Count; $index++) {
        $config = Join-Path $repo $configs[$index]
        Write-Host (
            "PHASE {0}/{1}: {2}" -f
                ($index + 1), $configs.Count, $configs[$index]
        )
        $arguments = @(
            $runner,
            '--config', $config,
            '--parallelism', $Parallelism
        )
        if ($DryRun) {
            $arguments += '--dry-run'
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Model comparison phase exited with code $LASTEXITCODE."
        }
    }

    if (-not $DryRun) {
        & $python (Join-Path $repo 'scripts\combine-final-model-comparison.py')
        if ($LASTEXITCODE -ne 0) {
            throw "Comparison summary exited with code $LASTEXITCODE."
        }
    }
} finally {
    if (-not $DryRun) {
        [void][ModelComparisonExecutionState]::SetThreadExecutionState(
            [uint32]2147483648
        )
    }
}

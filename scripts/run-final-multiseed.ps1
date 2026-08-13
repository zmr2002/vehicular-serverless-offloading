[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$Parallelism = 4,
    [switch]$Reset,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$config = Join-Path $repo 'configs\final-multiseed.toml'
$output = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\final-multiseed')
)
$expectedOutput = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\final-multiseed')
)
if (-not $output.Equals(
    $expectedOutput,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Refusing to use unexpected output path: $output"
}

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

if ($Reset -and -not $DryRun -and (Test-Path -LiteralPath $output)) {
    $resolvedOutput = (Resolve-Path -LiteralPath $output).Path
    if (-not $resolvedOutput.Equals(
        $expectedOutput,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove unexpected output path: $resolvedOutput"
    }
    Write-Host "Removing previous final experiment: $resolvedOutput"
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
$freeGiB = $drive.AvailableFreeSpace / 1GB
$reserveGiB = 12.0
$upperBoundGiB = 3.0
$requiredGiB = $reserveGiB + $upperBoundGiB
if (-not $DryRun -and $freeGiB -lt $requiredGiB) {
    throw (
        "Only {0:N2} GiB is free on {1}; at least {2:N2} GiB is required." -f
        $freeGiB, $drive.Name, $requiredGiB
    )
}
$diskMessage = (
    "Disk free before run: {0:N2} GiB. Estimated result upper bound: " +
    "{1:N2} GiB; protected reserve: {2:N2} GiB."
) -f $freeGiB, $upperBoundGiB, $reserveGiB
Write-Host $diskMessage
Write-Host (
    'The pipeline is resumable. Training task rows are disabled; evaluation ' +
    'task rows use a 0.1% sample.'
)

if (-not ('FinalExperimentExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class FinalExperimentExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

$arguments = @(
    (Join-Path $repo 'scripts\run-final-multiseed.py'),
    '--config', $config,
    '--parallelism', $Parallelism
)
if ($DryRun) {
    $arguments += '--dry-run'
}

try {
    if (-not $DryRun) {
        [void][FinalExperimentExecutionState]::SetThreadExecutionState(
            [uint32]2147483649
        )
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Final multi-seed pipeline exited with code $LASTEXITCODE."
    }
}
finally {
    if (-not $DryRun) {
        [void][FinalExperimentExecutionState]::SetThreadExecutionState(
            [uint32]2147483648
        )
    }
}

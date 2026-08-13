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
$config = Join-Path $repo 'configs\hybrid-seed-stability.toml'
$relativeOutput = 'results/verified/hybrid-seed-stability'
$output = [System.IO.Path]::GetFullPath((Join-Path $repo $relativeOutput))
$expectedOutput = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\hybrid-seed-stability')
)
if (-not $output.Equals($expectedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
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
    throw 'Commit or remove working-tree changes before running the stability pipeline.'
}

if ($Reset -and -not $DryRun -and (Test-Path -LiteralPath $output)) {
    $resolvedOutput = (Resolve-Path -LiteralPath $output).Path
    if (-not $resolvedOutput.Equals(
        $expectedOutput,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove unexpected output path: $resolvedOutput"
    }
    Write-Host "Removing previous stability results: $resolvedOutput"
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
$freeGiB = $drive.AvailableFreeSpace / 1GB
if (-not $DryRun -and $freeGiB -lt 20.0) {
    throw ((
        "Only {0:N2} GiB is free on {1}; at least 20 GiB is required " +
        "before starting the sampled stability pipeline."
    ) -f $freeGiB, $drive.Name)
}
Write-Host ((
    "Disk free before run: {0:N2} GiB. " +
    "Training details are disabled; validation details are sampled; " +
    "selected diagnostics have a 6 GiB raw-data budget."
) -f $freeGiB)

$arguments = @(
    (Join-Path $repo 'scripts\run-hybrid-seed-stability.py'),
    '--config', $config,
    '--parallelism', $Parallelism
)
if ($DryRun) {
    $arguments += '--dry-run'
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid seed stability pipeline exited with code $LASTEXITCODE."
}

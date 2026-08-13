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
$config = Join-Path $repo 'configs\paper-follower-game.toml'
$baseline = Join-Path $repo 'results\verified\adaptive-gate-paper-scale'
$replications = Join-Path $repo 'results\verified\adaptive-gate-reproducibility'
$output = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\adaptive-gate-cross-evaluation')
)
$expectedOutput = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\adaptive-gate-cross-evaluation')
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
    throw 'Commit or remove working-tree changes before running the cross-evaluation pipeline.'
}

if ($Reset -and -not $DryRun -and (Test-Path -LiteralPath $output)) {
    Write-Host "Removing previous cross-evaluation results: $output"
    Remove-Item -LiteralPath $output -Recurse -Force
}

if (-not $DryRun) {
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
    $freeGiB = $drive.AvailableFreeSpace / 1GB
    if ($freeGiB -lt 22.0) {
        throw (
            "Only {0:N2} GiB is free on {1}; at least 22 GiB is required." -f
            $freeGiB,
            $drive.Name
        )
    }
    Write-Host (
        "Disk free before run: {0:N2} GiB. Running 12 off-diagonal Hybrid evaluations." -f
        $freeGiB
    )
}

$arguments = @(
    (Join-Path $repo 'scripts\run-hybrid-cross-evaluation.py'),
    '--config', $config,
    '--source', $baseline,
    '--source', $replications,
    '--output-dir', $output,
    '--parallelism', $Parallelism
)
if ($DryRun) {
    $arguments += '--dry-run'
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid cross-evaluation exited with code $LASTEXITCODE."
}

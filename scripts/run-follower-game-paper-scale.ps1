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
$config = Join-Path $repo 'configs\training-evaluation-follower-game-paper-scale.toml'

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
    throw 'Commit or remove working-tree changes before running the resumable pipeline.'
}

$arguments = @(
    (Join-Path $repo 'scripts\run-training-evaluation.py'),
    '--config', $config,
    '--parallelism', $Parallelism
)
if ($DryRun) {
    $arguments += '--dry-run'
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Follower-game pipeline exited with code $LASTEXITCODE."
}

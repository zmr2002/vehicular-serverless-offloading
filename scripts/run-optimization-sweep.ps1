[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Config = 'configs\speed-optimization-sweep.toml',
    [string]$ResultRoot = 'results\verified\speed-optimization-sweep',
    [int]$ExpectedRuns = 14
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$configPath = (Resolve-Path -LiteralPath (Join-Path $repo $Config)).Path
$resultRoot = Join-Path $repo $ResultRoot
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$sessionDir = Join-Path $resultRoot ("driver-$stamp")
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $repo.Replace('\', '/')

$startedAt = Get-Date
$commit = (& git -C $repo rev-parse HEAD).Trim()
$status = @(& git -C $repo status --porcelain)
$metadata = [ordered]@{
    started_at = $startedAt.ToString('o')
    repository = $repo
    config = $configPath
    python = $python
    sumo_home = $sumo
    git_commit = $commit
    git_dirty = ($status.Count -gt 0)
    git_status = $status
    expected_runs = $ExpectedRuns
}
$metadataPath = Join-Path $sessionDir 'run-metadata.json'
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8

Write-Host "Starting $ExpectedRuns resumable optimization runs."
Write-Host "Diagnostics: $sessionDir"
$exitCode = 1
try {
    & $python (Join-Path $repo 'scripts\run-optimization-profiles.py') --config $configPath
    $exitCode = $LASTEXITCODE
}
finally {
    $finishedAt = Get-Date
    $metadata.finished_at = $finishedAt.ToString('o')
    $metadata.elapsed_seconds = ($finishedAt - $startedAt).TotalSeconds
    $metadata.exit_code = $exitCode
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8
}
if ($exitCode -ne 0) {
    throw "Optimization sweep exited with code $exitCode. Completed rows can be resumed."
}
Write-Host "Optimization sweep completed: $resultRoot\optimization-results.csv"

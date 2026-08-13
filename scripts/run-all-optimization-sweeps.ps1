[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' })
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $PSScriptRoot 'run-optimization-sweep.ps1'
$resultRoot = Join-Path $repo 'results\verified\combined-optimization-sweep'
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$sessionDir = Join-Path $resultRoot ("driver-$stamp")
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
$logPath = Join-Path $sessionDir 'runner.log'

$metadata = [ordered]@{
    started_at = (Get-Date).ToString('o')
    repository = $repo
    order = @('speed', 'result')
    speed_runs = 14
    result_runs = 16
    status = 'running'
}
$metadataPath = Join-Path $sessionDir 'run-metadata.json'
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8

try {
    Write-Host 'Phase 1/2: speed optimization (14 runs, physical model unchanged).'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $runner `
        -PythonExe $PythonExe `
        -SumoHome $SumoHome `
        -Config 'configs\speed-optimization-sweep.toml' `
        -ResultRoot 'results\verified\speed-optimization-sweep' `
        -ExpectedRuns 14 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Speed sweep exited with code $LASTEXITCODE" }

    Write-Host 'Phase 2/2: channel and cloud sensitivity (16 runs, workload unchanged).'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $runner `
        -PythonExe $PythonExe `
        -SumoHome $SumoHome `
        -Config 'configs\result-optimization-sweep.toml' `
        -ResultRoot 'results\verified\result-optimization-sweep' `
        -ExpectedRuns 16 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Result sweep exited with code $LASTEXITCODE" }

    & $PythonExe (Join-Path $repo 'scripts\summarize-optimization-sweeps.py') `
        --speed (Join-Path $repo 'results\verified\speed-optimization-sweep\optimization-results.csv') `
        --result (Join-Path $repo 'results\verified\result-optimization-sweep\optimization-results.csv') `
        --output (Join-Path $resultRoot 'screening-summary.md')
    if ($LASTEXITCODE -ne 0) { throw "Summary generation exited with code $LASTEXITCODE" }

    $metadata.status = 'completed'
}
catch {
    $metadata.status = 'failed'
    $metadata.error = $_.Exception.Message
    throw
}
finally {
    $metadata.finished_at = (Get-Date).ToString('o')
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8
}

Write-Host 'All optimization sweeps completed.'
Write-Host "Speed results: $repo\results\verified\speed-optimization-sweep\optimization-results.csv"
Write-Host "Result results: $repo\results\verified\result-optimization-sweep\optimization-results.csv"
Write-Host "Combined log: $logPath"
Write-Host "Summary: $resultRoot\screening-summary.md"

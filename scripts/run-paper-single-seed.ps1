[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Config = 'configs\paper-single-seed.toml',
    [string]$ResultSubdir = 'paper-single-seed',
    [int]$ExpectedRuns = 15,
    [int]$SampleIntervalSeconds = 5
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$configPath = (Resolve-Path -LiteralPath (Join-Path $repo $Config)).Path
if ([System.IO.Path]::IsPathRooted($ResultSubdir) -or $ResultSubdir -match '(^|[\\/])\.\.([\\/]|$)') {
    throw 'ResultSubdir must be a relative child path without parent traversal.'
}
if ($ExpectedRuns -le 0 -or $SampleIntervalSeconds -le 0) {
    throw 'ExpectedRuns and SampleIntervalSeconds must be positive.'
}
$resultRoot = Join-Path (Join-Path $repo 'results\verified') $ResultSubdir
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$sessionDir = Join-Path $resultRoot ("driver-$stamp")
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path
# Trust only this exact repository for this process tree; do not mutate the
# user's global Git configuration.
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $repo.Replace('\', '/')
$startedAt = Get-Date
$commitOutput = @(& git -C $repo rev-parse HEAD 2>&1)
if ($LASTEXITCODE -ne 0 -or $commitOutput.Count -eq 0) {
    throw "Unable to read the repository commit: $($commitOutput -join [Environment]::NewLine)"
}
$commit = ([string]$commitOutput[0]).Trim()
$status = @(& git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to read the repository working-tree status.'
}
$metadata = [ordered]@{
    started_at = $startedAt.ToString('o')
    repository = $repo
    config = $configPath
    python = $python
    sumo_home = $sumo
    git_commit = $commit
    git_dirty = ($status.Count -gt 0)
    git_status = $status
    sample_interval_seconds = $SampleIntervalSeconds
    logical_processors = $env:NUMBER_OF_PROCESSORS
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $sessionDir 'run-metadata.json') -Encoding utf8

$resourcePath = Join-Path $sessionDir 'resource-samples.csv'
$resourceWriter = [System.IO.StreamWriter]::new($resourcePath, $false, [System.Text.UTF8Encoding]::new($false))
$resourceWriter.WriteLine('timestamp,elapsed_s,python_processes,python_cpu_s,python_working_set_mb,python_private_mb,sumo_processes,sumo_cpu_s,sumo_working_set_mb,total_working_set_mb,completed_runs')
$stdoutPath = Join-Path $sessionDir 'console.stdout.log'
$stderrPath = Join-Path $sessionDir 'console.stderr.log'
$monitorErrorPath = Join-Path $sessionDir 'monitor-errors.log'
$progressPath = Join-Path $resultRoot 'matrix-progress.csv'

$arguments = @('-m', 'vehicular_offloading', 'experiment', '--config', $configPath)
$runner = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru -WindowStyle Hidden

Write-Host "Started paper matrix runner PID $($runner.Id)."
Write-Host "Session diagnostics: $sessionDir"
$lastCompleted = -1
try {
    while ($true) {
        try {
            $runner.Refresh()
            $pythonProcesses = @(
                Get-Process -Name 'python', 'pythonw' -ErrorAction SilentlyContinue |
                    Where-Object { $_.StartTime -ge $startedAt }
            )
            if ($runner.HasExited -and $pythonProcesses.Count -eq 0) { break }
            $elapsed = ((Get-Date) - $startedAt).TotalSeconds
            $completed = 0
            if ((Test-Path -LiteralPath $progressPath) -and
                (Get-Item -LiteralPath $progressPath).LastWriteTime -ge $startedAt) {
                try { $completed = @(Import-Csv -LiteralPath $progressPath).Count } catch { $completed = 0 }
            }
            if ($completed -ne $lastCompleted) {
                Write-Host ("[{0}] Completed groups: {1}/{2}" -f (Get-Date -Format 'HH:mm:ss'), $completed, $ExpectedRuns)
                $lastCompleted = $completed
            }

            $sumoProcesses = @(
                Get-Process -Name 'sumo', 'sumo-gui' -ErrorAction SilentlyContinue |
                    Where-Object { $_.StartTime -ge $startedAt }
            )
            $pythonCpu = ($pythonProcesses | Measure-Object -Property CPU -Sum).Sum
            $pythonWorking = ($pythonProcesses | Measure-Object -Property WorkingSet64 -Sum).Sum
            $pythonPrivate = ($pythonProcesses | Measure-Object -Property PrivateMemorySize64 -Sum).Sum
            $sumoCpu = ($sumoProcesses | Measure-Object -Property CPU -Sum).Sum
            $sumoWorking = ($sumoProcesses | Measure-Object -Property WorkingSet64 -Sum).Sum
            foreach ($name in @('pythonCpu', 'pythonWorking', 'pythonPrivate', 'sumoCpu', 'sumoWorking')) {
                if ($null -eq (Get-Variable -Name $name -ValueOnly)) { Set-Variable -Name $name -Value 0 }
            }
            $totalWorking = $pythonWorking + $sumoWorking
            $resourceWriter.WriteLine((
                '{0},{1:F3},{2},{3:F3},{4:F3},{5:F3},{6},{7:F3},{8:F3},{9:F3},{10}' -f
                (Get-Date).ToString('o'), $elapsed, $pythonProcesses.Count, $pythonCpu,
                ($pythonWorking / 1MB), ($pythonPrivate / 1MB), $sumoProcesses.Count,
                $sumoCpu, ($sumoWorking / 1MB), ($totalWorking / 1MB), $completed
            ))
            $resourceWriter.Flush()
        }
        catch {
            Add-Content -LiteralPath $monitorErrorPath -Value (
                '[{0}] {1}' -f (Get-Date).ToString('o'), $_.Exception.Message
            )
        }
        Start-Sleep -Seconds $SampleIntervalSeconds
    }
    $runner.WaitForExit()
}
finally {
    $resourceWriter.Dispose()
    if (-not $runner.HasExited) {
        Stop-Process -Id $runner.Id -Force
    }
}

$finishedAt = Get-Date
$metadata.finished_at = $finishedAt.ToString('o')
$metadata.elapsed_seconds = ($finishedAt - $startedAt).TotalSeconds
$metadata.exit_code = $runner.ExitCode
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $sessionDir 'run-metadata.json') -Encoding utf8

if ($runner.ExitCode -ne 0) {
    Write-Error "Matrix runner exited with code $($runner.ExitCode). See $stderrPath"
}

$completedRuns = 0
if (Test-Path -LiteralPath $progressPath) {
    $completedRuns = @(Import-Csv -LiteralPath $progressPath).Count
}
if ($completedRuns -ne $ExpectedRuns) {
    Write-Error "Matrix runner stopped after $completedRuns/$ExpectedRuns completed groups. See $sessionDir"
}

$detail = Get-ChildItem -LiteralPath $resultRoot -Filter 'matrix-detail-*.csv' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$summary = Get-ChildItem -LiteralPath $resultRoot -Filter 'matrix-summary-*.csv' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "All $ExpectedRuns groups completed successfully."
Write-Host "Detail: $($detail.FullName)"
Write-Host "Summary: $($summary.FullName)"
Write-Host "Diagnostics: $sessionDir"

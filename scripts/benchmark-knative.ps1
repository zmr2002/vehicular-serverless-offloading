[CmdletBinding()]
param(
    [string]$Endpoint,
    [string]$Profile = 'knative',
    [string]$PythonExe = 'python',
    [int]$ScaleToZeroTimeoutSeconds = 600,
    [int]$RequestsPerLevel = 100,
    [int]$WorkUnits = 250000,
    [int]$PollIntervalMilliseconds = 500
)

$ErrorActionPreference = 'Stop'

foreach ($directory in @(
    'C:\Program Files\Kubernetes\Minikube',
    (Join-Path $env:LOCALAPPDATA 'Programs\Knative')
)) {
    if ((Test-Path -LiteralPath $directory) -and
        (($env:Path -split ';') -notcontains $directory)) {
        $env:Path = "$directory;$env:Path"
    }
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Endpoint) {
    $Endpoint = kubectl --context $Profile get kservice vehicular-task-function -o jsonpath='{.status.url}'
}
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Endpoint)) {
    throw 'Could not resolve the Knative Service endpoint.'
}

$endpointUri = [Uri]$Endpoint
$endpointPort = if ($endpointUri.IsDefaultPort) {
    if ($endpointUri.Scheme -eq 'https') { 443 } else { 80 }
} else {
    $endpointUri.Port
}
$transport = [System.Net.Sockets.TcpClient]::new()
try {
    $connected = $transport.ConnectAsync(
        $endpointUri.DnsSafeHost,
        $endpointPort
    ).Wait(3000)
    if (-not $connected -or -not $transport.Connected) {
        throw 'connection timed out'
    }
} catch {
    throw (
        "Knative endpoint transport is unavailable at " +
        "$($endpointUri.DnsSafeHost):$endpointPort. Start: " +
        "minikube tunnel --profile $Profile"
    )
} finally {
    $transport.Dispose()
}

$deadline = (Get-Date).AddSeconds($ScaleToZeroTimeoutSeconds)
do {
    $pods = @(kubectl --context $Profile get pods -l serving.knative.dev/service=vehicular-task-function -o name 2>$null)
    if ($pods.Count -eq 0) { break }
    Start-Sleep -Seconds 5
} while ((Get-Date) -lt $deadline)
if ($pods.Count -ne 0) { throw 'Service did not scale to zero before the cold-start benchmark.' }

$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $repo "results\verified\serverless\$timestamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$timelinePath = Join-Path $runDir 'scaling-timeline.csv'
$stdoutPath = Join-Path $runDir 'benchmark.stdout.log'
$stderrPath = Join-Path $runDir 'benchmark.stderr.log'
$timeline = [System.Collections.Generic.List[object]]::new()

$env:PYTHONPATH = Join-Path $repo 'src'
$python = if (Test-Path -LiteralPath $PythonExe) {
    (Resolve-Path -LiteralPath $PythonExe).Path
} else {
    (Get-Command $PythonExe -ErrorAction Stop).Source
}
$arguments = @(
    '-m',
    'vehicular_offloading',
    'serverless-benchmark',
    '--endpoint',
    $Endpoint,
    '--output-dir',
    $runDir,
    '--requests',
    $RequestsPerLevel,
    '--work-units',
    $WorkUnits
)
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

do {
    $podList = kubectl --context $Profile `
        get pods `
        -l serving.knative.dev/service=vehicular-task-function `
        -o json | ConvertFrom-Json
    $readyPods = @(
        $podList.items |
        Where-Object {
            @($_.status.conditions | Where-Object {
                $_.type -eq 'Ready' -and $_.status -eq 'True'
            }).Count -gt 0
        }
    ).Count
    $timeline.Add([pscustomobject]@{
        timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
        phase = 'benchmark'
        pod_count = @($podList.items).Count
        ready_pod_count = $readyPods
        pod_names = (@($podList.items.metadata.name) -join ';')
    })
    Start-Sleep -Milliseconds $PollIntervalMilliseconds
    $process.Refresh()
} while (-not $process.HasExited)

$process.WaitForExit()
$process.Refresh()
$resultFiles = @(Get-ChildItem -LiteralPath $runDir -Filter 'serverless-benchmark-*.json')
$benchmarkFailed = (
    ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) -or
    $resultFiles.Count -eq 0
)
if ($benchmarkFailed) {
    $exitCode = if ($null -eq $process.ExitCode) { 'unavailable' } else { $process.ExitCode }
    throw "Serverless benchmark failed with exit code $exitCode. See $stderrPath"
}

$deadline = (Get-Date).AddSeconds($ScaleToZeroTimeoutSeconds)
do {
    $podList = kubectl --context $Profile `
        get pods `
        -l serving.knative.dev/service=vehicular-task-function `
        -o json | ConvertFrom-Json
    $timeline.Add([pscustomobject]@{
        timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
        phase = 'scale_down'
        pod_count = @($podList.items).Count
        ready_pod_count = @(
            $podList.items |
            Where-Object {
                @($_.status.conditions | Where-Object {
                    $_.type -eq 'Ready' -and $_.status -eq 'True'
                }).Count -gt 0
            }
        ).Count
        pod_names = (@($podList.items.metadata.name) -join ';')
    })
    if (@($podList.items).Count -eq 0) { break }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

$timeline | Export-Csv -LiteralPath $timelinePath -NoTypeInformation -Encoding utf8
if (@($podList.items).Count -ne 0) {
    throw "Service did not scale back to zero. See $timelinePath"
}

Write-Host "Serverless benchmark completed: $runDir"

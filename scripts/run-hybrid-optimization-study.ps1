[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Profile = 'knative',
    [int]$Parallelism = 6,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipServerless,
    [switch]$SkipBenchmark,
    [switch]$SkipLegacyServerless,
    [switch]$ServerlessOnly
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$config = Join-Path $repo 'configs\hybrid-optimization-study.toml'
$output = Join-Path $repo 'results\verified\hybrid-optimization-study'
$statePath = Join-Path $output 'study-state.json'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$driverStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$transcript = Join-Path $output "driver-$driverStamp.log"

if ($ServerlessOnly -and ($PreflightOnly -or $Smoke -or $SkipServerless)) {
    throw '-ServerlessOnly cannot be combined with PreflightOnly, Smoke, or SkipServerless.'
}

function Test-KnativeEndpoint {
    param([Parameter(Mandatory)][string]$Endpoint)
    try {
        $health = $Endpoint.TrimEnd('/') + '/healthz'
        $response = Invoke-WebRequest `
            -Uri $health `
            -UseBasicParsing `
            -TimeoutSec 5 `
            -ErrorAction Stop
        return [int]$response.StatusCode -eq 200
    } catch {
        return $false
    }
}

foreach ($directory in @(
    'C:\Program Files\Kubernetes\Minikube',
    (Join-Path $env:LOCALAPPDATA 'Programs\Knative')
)) {
    if ((Test-Path -LiteralPath $directory) -and
        (($env:Path -split ';') -notcontains $directory)) {
        $env:Path = "$directory;$env:Path"
    }
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
if (-not $PreflightOnly -and $status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before the resumable study.'
}

& $python -c 'import requests, torch, traci'
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python environment lacks required packages: $python"
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
$freeGiB = $drive.AvailableFreeSpace / 1GB
$reserveGiB = 12.0
$upperBoundGiB = 5.0
if (-not $PreflightOnly -and $freeGiB -lt ($reserveGiB + $upperBoundGiB)) {
    throw (
        'Only {0:N2} GiB is free on {1}; at least {2:N2} GiB is required.' -f
        $freeGiB, $drive.Name, ($reserveGiB + $upperBoundGiB)
    )
}
Write-Host ((
    'Disk free before run: {0:N2} GiB. Result ceiling: {1:N2} GiB; ' +
    'protected reserve: {2:N2} GiB.'
) -f $freeGiB, $upperBoundGiB, $reserveGiB)
Write-Host 'The study is resumable and uses six worker processes by default.'
Write-Host 'Training task rows are disabled; evaluation retains a 0.1% sample.'
Write-Host (
    'The analytical ceiling is 307 simulations before semantic duplicate ' +
    'removal; screening stops no stages early based on a favorable result.'
)

if (-not ('HybridStudyExecutionState' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class HybridStudyExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
}

$arguments = @(
    (Join-Path $repo 'scripts\run-hybrid-optimization-study.py'),
    '--config', $config,
    '--parallelism', $Parallelism
)
if ($PreflightOnly) { $arguments += '--dry-run' }
if ($Smoke) { $arguments += '--smoke' }

$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $transcript -Append | Out-Null
    $transcriptStarted = $true
    if (-not $PreflightOnly) {
        [void][HybridStudyExecutionState]::SetThreadExecutionState(
            [uint32]2147483649
        )
    }
    if ($ServerlessOnly) {
        if (-not (Test-Path -LiteralPath $statePath)) {
            throw "Completed analytical study state is missing: $statePath"
        }
        $savedState = Get-Content -Raw -Encoding utf8 $statePath | ConvertFrom-Json
        if ($savedState.status -ne 'analytical_complete') {
            throw (
                'ServerlessOnly requires status=analytical_complete; found ' +
                "'$($savedState.status)'."
            )
        }
        Write-Host 'Reusing the completed analytical study; starting at Serverless validation.'
    } else {
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Hybrid optimization study exited with code $LASTEXITCODE."
        }
    }

    if ($PreflightOnly -or $Smoke -or $SkipServerless) {
        if ($SkipServerless) {
            Write-Host 'Serverless validation was explicitly skipped.'
        }
        return
    }
    if (-not (Test-Path -LiteralPath $statePath)) {
        throw "Study state was not generated: $statePath"
    }
    $state = Get-Content -Raw -Encoding utf8 $statePath | ConvertFrom-Json
    foreach ($command in @('docker', 'minikube', 'kubectl')) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required serverless command is missing: $command"
        }
    }
    $osType = docker info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0 -or $osType.Trim() -ne 'linux') {
        throw 'Docker Desktop must be running with Linux containers.'
    }

    minikube status --profile $Profile *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Starting Minikube profile '$Profile'."
        minikube start --profile $Profile
        if ($LASTEXITCODE -ne 0) {
            throw "Could not start Minikube profile '$Profile'."
        }
    }
    & (Join-Path $PSScriptRoot 'deploy-knative.ps1') -Profile $Profile

    $endpoint = kubectl --context $Profile `
        get kservice vehicular-task-function `
        -o jsonpath='{.status.url}'
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($endpoint)) {
        throw 'Could not resolve the Knative Service endpoint.'
    }

    $externalIp = kubectl --context $Profile `
        --namespace kourier-system `
        get service kourier `
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    $endpointReachable = Test-KnativeEndpoint -Endpoint $endpoint
    if ([string]::IsNullOrWhiteSpace($externalIp) -or -not $endpointReachable) {
        Write-Host (
            'Starting Minikube tunnel because the Knative endpoint is not ' +
            'reachable from the host.'
        )
        $tunnel = Start-Process `
            -FilePath (Get-Command minikube).Source `
            -ArgumentList @('tunnel', '--profile', $Profile) `
            -WindowStyle Hidden `
            -PassThru
        $deadline = (Get-Date).AddMinutes(2)
        do {
            Start-Sleep -Seconds 2
            if ($tunnel.HasExited) {
                throw (
                    'Minikube tunnel exited before Kourier received an IP. ' +
                    'Run this script from an elevated PowerShell window.'
                )
            }
            $externalIp = kubectl --context $Profile `
                --namespace kourier-system `
                get service kourier `
                -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
            $endpointReachable = Test-KnativeEndpoint -Endpoint $endpoint
        } until (
            (
                -not [string]::IsNullOrWhiteSpace($externalIp) -and
                $endpointReachable
            ) -or
            (Get-Date) -ge $deadline
        )
        if (-not $endpointReachable) {
            throw (
                'The Knative endpoint remained unreachable after starting ' +
                'Minikube tunnel. Run from an elevated PowerShell window.'
            )
        }
    }

    if (-not $SkipBenchmark) {
        Write-Host 'Running cold/warm and concurrency 1/10/50 benchmark.'
        & (Join-Path $PSScriptRoot 'benchmark-knative.ps1') `
            -Profile $Profile `
            -PythonExe $python `
            -RequestsPerLevel 100 `
            -WorkUnits 250000
    }

    if (-not $SkipLegacyServerless) {
        Write-Host 'Validating the frozen pre-serverless optimized Hybrid model.'
        & (Join-Path $PSScriptRoot 'run-knative-validation.ps1') `
            -PythonExe $python `
            -SumoHome $sumo `
            -Profile $Profile `
            -Config (Join-Path $repo 'configs\knative-pre-serverless-three-seed.toml')
    }

    if (-not $state.serverless_ready) {
        Write-Host (
            'The new analytical winner did not pass the predeclared gate. ' +
            'Its Knative validation was not started; the platform benchmark ' +
            'and frozen-model control remain available.'
        ) -ForegroundColor Yellow
        return
    }

    $serverlessConfig = Join-Path $output 'generated-configs\knative-final.toml'
    $serverlessOutput = Join-Path $output 'serverless'
    & $python `
        (Join-Path $repo 'scripts\prepare-final-serverless-config.py') `
        --final-config ([string]$state.final_pipeline) `
        --base-config ([string]$state.final_profile) `
        --output $serverlessConfig `
        --validation-output-dir $serverlessOutput `
        --analytical-parallelism $Parallelism
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not generate the selected-model Knative validation config.'
    }
    & (Join-Path $PSScriptRoot 'run-knative-validation.ps1') `
        -PythonExe $python `
        -SumoHome $sumo `
        -Profile $Profile `
        -Config $serverlessConfig
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
    if (-not $PreflightOnly) {
        [void][HybridStudyExecutionState]::SetThreadExecutionState(
            [uint32]2147483648
        )
    }
}

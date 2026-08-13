[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [string]$Profile = 'knative',
    [string]$Config = 'configs\knative-validation.toml',
    [int[]]$Steps,
    [int[]]$VehicleCounts,
    [string]$Checkpoint,
    [string]$ResumeSession,
    [switch]$PreflightOnly
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
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$configCandidate = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $repo $Config
}
$configPath = (Resolve-Path $configCandidate).Path
$runner = Join-Path $PSScriptRoot 'run-knative-validation.py'

foreach ($command in @('docker', 'minikube', 'kubectl')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is not installed or not on PATH: $command"
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
if ($status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before running resumable validation.'
}
& $python -c 'import requests, torch, traci'
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python environment lacks required packages: $python"
}

$osType = docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0 -or $osType.Trim() -ne 'linux') {
    throw 'Docker Desktop must be running with Linux containers.'
}
minikube status --profile $Profile
if ($LASTEXITCODE -ne 0) {
    throw "Minikube profile '$Profile' is not running."
}

kubectl --context $Profile `
    wait `
    --for=condition=Ready `
    kservice/vehicular-task-function `
    --timeout=10s
if ($LASTEXITCODE -ne 0) {
    throw 'Knative Service vehicular-task-function is not ready.'
}
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
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($externalIp)) {
    throw "Kourier has no external IP. Start: minikube tunnel --profile $Profile"
}
try {
    $health = $endpoint.TrimEnd('/') + '/healthz'
    $response = Invoke-WebRequest `
        -Uri $health `
        -UseBasicParsing `
        -TimeoutSec 10 `
        -ErrorAction Stop
    if ([int]$response.StatusCode -ne 200) {
        throw "Unexpected health status: $($response.StatusCode)"
    }
} catch {
    throw (
        "Knative endpoint is not reachable even though Kourier reports " +
        "external IP '$externalIp'. Restart: minikube tunnel --profile $Profile"
    )
}

$arguments = @(
    $runner,
    '--config',
    $configPath,
    '--endpoint',
    $endpoint,
    '--profile',
    $Profile
)
if ($Steps.Count -gt 0) {
    $arguments += '--steps'
    $arguments += @($Steps | ForEach-Object { $_.ToString() })
}
if ($VehicleCounts.Count -gt 0) {
    $arguments += '--vehicles'
    $arguments += @($VehicleCounts | ForEach-Object { $_.ToString() })
}
if ($Checkpoint) {
    $arguments += @('--checkpoint', $Checkpoint)
}
if ($ResumeSession) {
    $arguments += @('--session', $ResumeSession)
}
if ($PreflightOnly) {
    $arguments += '--preflight-only'
}

Write-Host "Knative endpoint: $endpoint"
Write-Host 'The analytical and live runs use identical mobility, tasks, seed, and policy.'
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Knative validation exited with code $LASTEXITCODE."
}

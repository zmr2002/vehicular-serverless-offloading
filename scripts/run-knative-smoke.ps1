[CmdletBinding()]
param(
    [string]$Profile = 'knative',
    [string]$Config = 'configs/smoke.toml',
    [string]$SimulatorImage = 'vehicular-offloading-simulator:0.1.0'
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
$results = Join-Path $repo 'results\verified'
$dockerfile = Join-Path $repo 'Dockerfile'

minikube status --profile $Profile
if ($LASTEXITCODE -ne 0) { throw "Minikube profile '$Profile' is not running." }

$endpoint = kubectl --context $Profile `
    get kservice vehicular-task-function `
    -o jsonpath='{.status.url}'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($endpoint)) {
    throw 'Could not resolve the Knative Service endpoint.'
}
$hostName = ([System.Uri]$endpoint).Host

$externalIp = kubectl --context $Profile `
    --namespace kourier-system `
    get service kourier `
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($externalIp)) {
    throw "Kourier has no external IP. Start: minikube tunnel --profile $Profile"
}

docker build `
    --provenance=false `
    --sbom=false `
    --tag $SimulatorImage `
    --file $dockerfile `
    $repo
if ($LASTEXITCODE -ne 0) { throw 'Building the simulator image failed.' }

docker run `
    --rm `
    --add-host "${hostName}:host-gateway" `
    --volume "${results}:/app/results/verified" `
    $SimulatorImage `
    simulate `
    --config $Config `
    --backend knative `
    --endpoint $endpoint
if ($LASTEXITCODE -ne 0) { throw 'The simulator-to-Knative smoke run failed.' }

[CmdletBinding()]
param(
    [string]$Profile = 'knative',
    [int]$Cpus = 4,
    [int]$MemoryMb = 6144,
    [string]$KubernetesVersion = '1.35.1',
    [switch]$InstallEventing
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

foreach ($command in @('docker', 'minikube', 'kubectl', 'kn')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is not installed or not on PATH: $command"
    }
}
if (-not (Get-Command 'kn-quickstart' -ErrorAction SilentlyContinue)) {
    throw 'The Knative quickstart plugin is not installed or not on PATH: kn-quickstart'
}

$osType = docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0 -or $osType.Trim() -ne 'linux') {
    throw 'Docker Desktop must be running with Linux containers.'
}

$profiles = minikube profile list -o json | ConvertFrom-Json
$profileExists = @($profiles.valid | ForEach-Object { $_.Name }) -contains $Profile
if ($profileExists) {
    Write-Host "Minikube profile '$Profile' already exists; starting it if necessary."
    minikube start --profile $Profile
    if ($LASTEXITCODE -ne 0) { throw "Could not start Minikube profile '$Profile'." }
} else {
    $quickstartArguments = @(
        'quickstart',
        'minikube',
        '--name',
        $Profile,
        '--kubernetes-version',
        $KubernetesVersion,
        '--install-serving'
    )
    if ($InstallEventing) {
        $quickstartArguments += '--install-eventing'
    }
    $quickstartArguments += @(
        '--',
        '--driver=docker',
        "--cpus=$Cpus",
        "--memory=$MemoryMb"
    )
    & kn @quickstartArguments
    if ($LASTEXITCODE -ne 0) { throw 'Knative quickstart failed.' }
}

kubectl --context $Profile wait `
    --namespace knative-serving `
    --for=condition=Available deployment/controller `
    --timeout=300s
if ($LASTEXITCODE -ne 0) { throw 'Knative Serving controller did not become available.' }

Write-Host "Knative Serving is ready in Minikube profile '$Profile'."
Write-Host "Keep this running in a second terminal: minikube tunnel --profile $Profile"

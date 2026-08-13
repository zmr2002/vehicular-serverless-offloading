[CmdletBinding()]
param(
    [string]$Profile = 'knative'
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
$dockerfile = Join-Path $repo 'serverless_function\Dockerfile'
$context = Join-Path $repo 'serverless_function'
$manifest = Join-Path $repo 'deploy\knative\service.yaml'
$image = 'dev.local/vehicular-task-function:0.1.1'

minikube status --profile $Profile
if ($LASTEXITCODE -ne 0) { throw "Minikube profile '$Profile' is not running." }
kubectl --context $Profile `
    --namespace knative-serving `
    wait `
    --for=condition=Available `
    deployment/webhook `
    --timeout=300s
if ($LASTEXITCODE -ne 0) { throw 'Knative admission webhook did not become available.' }
docker build `
    --provenance=false `
    --sbom=false `
    --tag $image `
    --file $dockerfile `
    $context
if ($LASTEXITCODE -ne 0) { throw 'Building the function image failed.' }
minikube -p $Profile image load --overwrite=true $image
if ($LASTEXITCODE -ne 0) { throw 'Loading the function image into Minikube failed.' }
$loadedImages = @(minikube -p $Profile image ls)
if ($loadedImages -notcontains $image) {
    throw "The function image is not present inside Minikube: $image"
}

$currentRegistries = kubectl --context $Profile `
    --namespace knative-serving `
    get configmap config-deployment `
    -o jsonpath='{.data.registries-skipping-tag-resolving}'
if ($LASTEXITCODE -ne 0) { throw 'Could not read the Knative deployment configuration.' }
$registries = @(
    @($currentRegistries -split ',') + 'dev.local' |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ } |
    Sort-Object -Unique
)
$patch = @{
    data = @{
        'registries-skipping-tag-resolving' = $registries -join ','
    }
} | ConvertTo-Json -Compress
$patchFile = [System.IO.Path]::GetTempFileName()
try {
    [System.IO.File]::WriteAllText(
        $patchFile,
        $patch,
        [System.Text.UTF8Encoding]::new($false)
    )
    $patched = $false
    foreach ($attempt in 1..6) {
        kubectl --context $Profile `
            --namespace knative-serving `
            patch configmap config-deployment `
            --type merge `
            --patch-file $patchFile
        if ($LASTEXITCODE -eq 0) {
            $patched = $true
            break
        }
        if ($attempt -lt 6) { Start-Sleep -Seconds 5 }
    }
    if (-not $patched) { throw 'Could not configure local Knative image resolution.' }
} finally {
    Remove-Item -LiteralPath $patchFile -Force -ErrorAction SilentlyContinue
}

kubectl --context $Profile apply -f $manifest
if ($LASTEXITCODE -ne 0) { throw 'Applying the Knative Service manifest failed.' }
kubectl --context $Profile wait --for=condition=Ready kservice/vehicular-task-function --timeout=300s
if ($LASTEXITCODE -ne 0) { throw 'The Knative Service did not become ready.' }
$url = kubectl --context $Profile get kservice vehicular-task-function -o jsonpath='{.status.url}'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($url)) {
    throw 'Could not resolve the Knative Service URL.'
}
Write-Host "Knative service URL: $url"

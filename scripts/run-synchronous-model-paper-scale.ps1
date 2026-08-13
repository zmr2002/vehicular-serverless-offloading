[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path

Write-Host 'Phase 1/2: validate the synchronized decision model.'
$invariantOutput = Join-Path $repo 'results\verified\synchronous-model-paper-scale\model-invariants.json'
& $python (Join-Path $repo 'scripts\analyze-model-invariants.py') `
    --config (Join-Path $repo 'configs\paper-improved.toml') `
    --output $invariantOutput
if ($LASTEXITCODE -ne 0) {
    throw "Model invariant checks exited with code $LASTEXITCODE."
}

Write-Host 'Phase 2/2: train and evaluate the paper-scale matrix.'
$runnerArguments = @(
    '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $repo 'scripts\run-training-evaluation-diagnostics.ps1'),
    '-PythonExe', $python,
    '-SumoHome', $sumo,
    '-Config', 'configs\training-evaluation-synchronous-paper-scale.toml'
)
if ($DryRun) {
    $runnerArguments += '-DryRun'
}
& powershell @runnerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Synchronous paper-scale pipeline exited with code $LASTEXITCODE."
}

Write-Host "Invariant report: $invariantOutput"

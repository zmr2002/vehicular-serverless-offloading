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

Write-Host 'Phase 1/3: unit and invariant checks.'
& $python -m unittest discover -s (Join-Path $repo 'tests') -v
if ($LASTEXITCODE -ne 0) {
    throw "Unit tests exited with code $LASTEXITCODE."
}

$invariantOutput = Join-Path $repo 'results\verified\reviewed-model-validation\model-invariants.json'
& $python (Join-Path $repo 'scripts\analyze-model-invariants.py') `
    --config (Join-Path $repo 'configs\paper-improved.toml') `
    --output $invariantOutput
if ($LASTEXITCODE -ne 0) {
    throw "Model invariant checks exited with code $LASTEXITCODE."
}

Write-Host 'Phase 2/3: validate the staged training/evaluation matrix.'
$runnerArguments = @(
    '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $repo 'scripts\run-training-evaluation-diagnostics.ps1'),
    '-PythonExe', $python,
    '-SumoHome', $sumo,
    '-Config', 'configs\training-evaluation-reviewed-validation.toml'
)
if ($DryRun) {
    $runnerArguments += '-DryRun'
}
& powershell @runnerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed validation pipeline exited with code $LASTEXITCODE."
}

Write-Host 'Phase 3/3: complete.'
Write-Host "Invariant report: $invariantOutput"

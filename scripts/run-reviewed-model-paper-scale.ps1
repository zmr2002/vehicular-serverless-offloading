[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' })
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo

& $python (Join-Path $repo 'scripts\analyze-model-invariants.py') `
    --config (Join-Path $repo 'configs\paper-improved.toml') `
    --output (Join-Path $repo 'results\verified\reviewed-model-paper-scale\model-invariants.json')
if ($LASTEXITCODE -ne 0) {
    throw "Model invariant checks exited with code $LASTEXITCODE."
}

& powershell -ExecutionPolicy Bypass `
    -File (Join-Path $repo 'scripts\run-training-evaluation-diagnostics.ps1') `
    -PythonExe $python `
    -SumoHome $sumo `
    -Config 'configs\training-evaluation-reviewed-paper-scale.toml'
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed paper-scale pipeline exited with code $LASTEXITCODE."
}

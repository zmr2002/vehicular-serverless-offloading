[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'),
    [string]$SumoHome = $(if ($env:SUMO_HOME) { $env:SUMO_HOME } else { Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\sumo' }),
    [int]$Parallelism = 4,
    [switch]$Reset,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$sumo = (Resolve-Path -LiteralPath $SumoHome).Path
$config = Join-Path $repo 'configs\training-evaluation-adaptive-gate-paper-scale.toml'
$relativeOutput = 'results/verified/adaptive-gate-reproducibility'
$output = [System.IO.Path]::GetFullPath((Join-Path $repo $relativeOutput))
$expectedOutput = [System.IO.Path]::GetFullPath(
    (Join-Path $repo 'results\verified\adaptive-gate-reproducibility')
)
if (-not $output.Equals($expectedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use unexpected output path: $output"
}

$env:PYTHONPATH = (Join-Path $repo 'src') + ';' + $repo
$env:SUMO_HOME = $sumo
$env:Path = (Join-Path $sumo 'bin') + ';' + $env:Path
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $repo.Replace('\', '/')

$status = @(& git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect the repository status.'
}
if (-not $DryRun -and $status.Count -gt 0) {
    throw 'Commit or remove working-tree changes before running the reproducibility pipeline.'
}

if ($Reset -and -not $DryRun -and (Test-Path -LiteralPath $output)) {
    Write-Host "Removing previous reproducibility results: $output"
    Remove-Item -LiteralPath $output -Recurse -Force
}

if (-not $DryRun) {
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($repo))
    $freeGiB = $drive.AvailableFreeSpace / 1GB
    $completedReplications = 0
    if (Test-Path -LiteralPath $output) {
        $completedReplications = @(
            Get-ChildItem -LiteralPath $output -Recurse `
                -Filter 'diagnostic-summary.json' -File
        ).Count
    }
    if ($completedReplications -gt 2) {
        throw (
            "Found $completedReplications completed sessions in $output. " +
            "Run again with -Reset to start a clean two-replication set."
        )
    }
    $remainingReplications = [Math]::Max(0, 2 - $completedReplications)
    $requiredFreeGiB = 12.0 + 8.0 * $remainingReplications
    if ($freeGiB -lt $requiredFreeGiB) {
        throw (
            (
                "Only {0:N2} GiB is free on {1}; {2} remaining full-detail " +
                "replication(s) require at least {3:N2} GiB before starting."
            ) -f $freeGiB, $drive.Name, $remainingReplications, $requiredFreeGiB
        )
    }
    Write-Host (
        (
            "Disk free before run: {0:N2} GiB. Existing seed 42 remains unchanged; " +
            "{1} full-detail replication(s) remain."
        ) -f $freeGiB, $remainingReplications
    )
}

$replications = @(
    @{ TrainingSeed = 31416; EvaluationSeed = 43 },
    @{ TrainingSeed = 31417; EvaluationSeed = 44 }
)

for ($index = 0; $index -lt $replications.Count; $index++) {
    $replication = $replications[$index]
    Write-Host (
        "REPLICATION {0}/{1}: training seed={2}, evaluation seed={3}" -f
        ($index + 1),
        $replications.Count,
        $replication.TrainingSeed,
        $replication.EvaluationSeed
    )
    $arguments = @(
        (Join-Path $repo 'scripts\run-training-evaluation.py'),
        '--config', $config,
        '--parallelism', $Parallelism,
        '--training-seed', $replication.TrainingSeed,
        '--evaluation-seed', $replication.EvaluationSeed,
        '--output-dir', $relativeOutput
    )
    if ($DryRun) {
        $arguments += '--dry-run'
    }

    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Replication {0} exited with code {1}." -f
            ($index + 1),
            $LASTEXITCODE
        )
    }
}

if (-not $DryRun) {
    $baseline = Join-Path $repo 'results\verified\adaptive-gate-paper-scale'
    & $python (Join-Path $repo 'scripts\summarize-reproducibility.py') `
        --input $baseline `
        --input $output `
        --output-dir $output `
        --expected-replications 3
    if ($LASTEXITCODE -ne 0) {
        throw "Reproducibility summary exited with code $LASTEXITCODE."
    }
    Write-Host "COMPLETE $output"
}

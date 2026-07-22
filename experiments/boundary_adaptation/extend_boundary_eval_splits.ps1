param(
    [int]$ValTargetEpisodes = 40,
    [int]$TestTargetEpisodes = 50,
    [int]$MaxAttemptsPerSplit = 180,
    [switch]$Background
)

# MuJoCo and Python may emit harmless warnings on stderr.
$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

if ($Background) {
    $LogRoot = Join-Path $ProjectRoot "logs"
    New-Item -ItemType Directory -Force $LogRoot | Out-Null
    $LauncherOut = Join-Path $LogRoot "extend_boundary_eval_splits.out.log"
    $LauncherErr = Join-Path $LogRoot "extend_boundary_eval_splits.err.log"
    $ChildArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-ValTargetEpisodes", "$ValTargetEpisodes",
        "-TestTargetEpisodes", "$TestTargetEpisodes",
        "-MaxAttemptsPerSplit", "$MaxAttemptsPerSplit"
    )
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $ChildArguments `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $LauncherOut `
        -RedirectStandardError $LauncherErr `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Boundary val/test extension started in background."
    Write-Host "PID=$($Process.Id)"
    Write-Host "Log: $LauncherOut"
    Write-Host "Error log: $LauncherErr"
    exit 0
}

$Collector = Join-Path $PSScriptRoot "collect_boundary_dataset.py"
$DataRoot = Join-Path $ProjectRoot (
    "datasets\aloha2-role-composition\raw_npz"
)
$Tasks = @(
    @{
        Name = "left_pick_place_after_right_push"
        SeedOffset = 0
    },
    @{
        Name = "right_pick_place_after_left_push"
        SeedOffset = 100000
    }
)
$Splits = @(
    @{
        Name = "val"
        Target = $ValTargetEpisodes
        Seed = 150000
    },
    @{
        Name = "primitive_test"
        Target = $TestTargetEpisodes
        Seed = 160000
    }
)

Push-Location $ProjectRoot
try {
    foreach ($Task in $Tasks) {
        foreach ($Split in $Splits) {
            $Output = Join-Path $DataRoot (
                "$($Split.Name)\$($Task.Name)"
            )
            New-Item -ItemType Directory -Force $Output | Out-Null
            $Existing = @(
                Get-ChildItem $Output -File -Filter "episode_*.npz"
            ).Count
            if ($Existing -ge $Split.Target) {
                Write-Host (
                    "Skipping $($Task.Name) $($Split.Name): " +
                    "existing=$Existing, target=$($Split.Target)"
                )
                continue
            }

            # Move each resumed invocation into a fresh deterministic seed
            # range so an interrupted run does not regenerate saved episodes.
            $Seed = $Split.Seed + $Task.SeedOffset + ($Existing * 1000)
            Write-Host ""
            Write-Host (
                "=== extending $($Task.Name) $($Split.Name): " +
                "existing=$Existing, target=$($Split.Target), seed=$Seed ==="
            )
            & $Python -u $Collector `
                --task $Task.Name `
                --episodes $Split.Target `
                --output $Output `
                --seed $Seed `
                --max-attempts $MaxAttemptsPerSplit `
                --resume
            $PythonExitCode = $LASTEXITCODE
            if ($PythonExitCode -ne 0) {
                throw (
                    "Extension failed for task=$($Task.Name), " +
                    "split=$($Split.Name), exit_code=$PythonExitCode"
                )
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host (
    "Both boundary tasks now have val=$ValTargetEpisodes and " +
    "primitive_test=$TestTargetEpisodes episodes."
)

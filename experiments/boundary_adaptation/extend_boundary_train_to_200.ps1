param(
    [int]$TargetEpisodes = 200,
    [int]$MaxAttemptsPerTask = 600,
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
    $LauncherOut = Join-Path $LogRoot "extend_boundary_train_to_200.out.log"
    $LauncherErr = Join-Path $LogRoot "extend_boundary_train_to_200.err.log"
    $ChildArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-TargetEpisodes", "$TargetEpisodes",
        "-MaxAttemptsPerTask", "$MaxAttemptsPerTask"
    )
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $ChildArguments `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $LauncherOut `
        -RedirectStandardError $LauncherErr `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Boundary train extension started in background."
    Write-Host "PID=$($Process.Id)"
    Write-Host "Log: $LauncherOut"
    Write-Host "Error log: $LauncherErr"
    exit 0
}

$Collector = Join-Path $PSScriptRoot "collect_boundary_dataset.py"
$TrainRoot = Join-Path $ProjectRoot (
    "datasets\aloha2-role-composition\raw_npz\train"
)
$Tasks = @(
    @{
        Name = "left_pick_place_after_right_push"
        Seed = 140000
    },
    @{
        Name = "right_pick_place_after_left_push"
        Seed = 240000
    }
)

Push-Location $ProjectRoot
try {
    foreach ($Task in $Tasks) {
        $Output = Join-Path $TrainRoot $Task.Name
        New-Item -ItemType Directory -Force $Output | Out-Null
        $Existing = @(
            Get-ChildItem $Output -File -Filter "episode_*.npz"
        ).Count
        Write-Host ""
        Write-Host (
            "=== extending $($Task.Name): existing=$Existing, " +
            "target=$TargetEpisodes, seed_start=$($Task.Seed) ==="
        )
        & $Python -u $Collector `
            --task $Task.Name `
            --episodes $TargetEpisodes `
            --output $Output `
            --seed $Task.Seed `
            --max-attempts $MaxAttemptsPerTask `
            --resume
        $PythonExitCode = $LASTEXITCODE
        if ($PythonExitCode -ne 0) {
            throw (
                "Extension failed for task=$($Task.Name), " +
                "exit_code=$PythonExitCode"
            )
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Both boundary training datasets reached $TargetEpisodes episodes."

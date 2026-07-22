param(
    [string]$Device = "cuda",
    [int]$Epochs = 100,
    [int]$NumWorkers = 2,
    [int]$Seed = 0,
    [switch]$Background
)

# Native Python libraries commonly write harmless warnings to stderr.
# Keep those warnings visible and decide success from Python's exit code.
$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

if ($Background) {
    $LogRoot = Join-Path $ProjectRoot "logs"
    New-Item -ItemType Directory -Force $LogRoot | Out-Null
    $LauncherOut = Join-Path $LogRoot (
        "language_act_symmetric_boundary_250.launcher.out.log"
    )
    $LauncherErr = Join-Path $LogRoot (
        "language_act_symmetric_boundary_250.launcher.err.log"
    )
    $ChildArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-Device", $Device,
        "-Epochs", "$Epochs",
        "-NumWorkers", "$NumWorkers",
        "-Seed", "$Seed"
    )
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $ChildArguments `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $LauncherOut `
        -RedirectStandardError $LauncherErr `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Symmetric-boundary ACT training started in background."
    Write-Host "PID=$($Process.Id)"
    Write-Host "Training log: logs\language_act_symmetric_boundary_250.log"
    Write-Host "Launcher error log: $LauncherErr"
    exit 0
}

$DataRoot = Join-Path $ProjectRoot "datasets\aloha2-role-composition\raw_npz"
$TrainTasks = @(
    "seen_lr",
    "left_tray_push",
    "right_tray_push",
    "left_pick_place_after_right_push",
    "right_pick_place_after_left_push"
)
$ValTasks = $TrainTasks

$Arguments = @("-u", "training\train_language_act.py")
foreach ($Task in $TrainTasks) {
    $Directory = Join-Path $DataRoot "train\$Task"
    if (-not (Test-Path $Directory)) {
        throw "Missing training directory: $Directory"
    }
    $Arguments += @("--train-dir", $Directory)
}
foreach ($Task in $ValTasks) {
    $Directory = Join-Path $DataRoot "val\$Task"
    if (-not (Test-Path $Directory)) {
        throw "Missing validation directory: $Directory"
    }
    $Arguments += @("--val-dir", $Directory)
}

$Output = Join-Path $ProjectRoot "checkpoints\language_act_symmetric_boundary_250"
$Log = Join-Path $ProjectRoot "logs\language_act_symmetric_boundary_250.log"
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null

$Arguments += @(
    "--max-train-episodes-per-task", "50",
    "--max-val-episodes-per-task", "10",
    "--task-balanced-sampling",
    "--output", $Output,
    "--chunk-size", "10",
    "--kl-weight", "10",
    "--batch-size", "10",
    "--learning-rate", "1e-4",
    "--epochs", "$Epochs",
    "--max-batches-per-epoch", "100",
    "--num-workers", "$NumWorkers",
    "--seed", "$Seed",
    "--device", $Device
)

Push-Location $ProjectRoot
try {
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log
    $PythonExitCode = $LASTEXITCODE
    if ($PythonExitCode -ne 0) {
        throw "Symmetric-boundary ACT training failed with exit code $PythonExitCode"
    }
}
finally {
    Pop-Location
}

Write-Host "Training completed: $Output"

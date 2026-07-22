param(
    [string]$Device = "cuda",
    [int]$Epochs = 100,
    [int]$NumWorkers = 2,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

$DataRoot = Join-Path $ProjectRoot "datasets\aloha2-role-composition\raw_npz"
$TrainTasks = @(
    "seen_lr",
    "left_tray_push",
    "right_tray_push",
    "left_pick_place",
    "right_pick_place",
    "left_pick_place_after_right_push"
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

$Output = Join-Path $ProjectRoot "checkpoints\language_act_boundary_balanced_300"
$Log = Join-Path $ProjectRoot "logs\language_act_boundary_balanced_300.log"
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null

$Arguments += @(
    "--max-train-episodes-per-task", "50",
    "--max-val-episodes-per-task", "10",
    "--task-balanced-sampling",
    "--output", $Output,
    "--chunk-size", "10",
    "--kl-weight", "10",
    "--batch-size", "12",
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
    if ($LASTEXITCODE -ne 0) {
        throw "Boundary ACT training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Training completed: $Output"

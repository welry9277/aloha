param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [ValidateSet("seen", "composition", "composition-worst", "all")]
    [string]$Model = "all",
    [ValidateSet("main", "primitives", "all")]
    [string]$Suite = "all",
    [int]$ExecuteActions = 10,
    [int]$MaxActions = 220,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

$models = @{
    seen = "checkpoints\language_act_seen_only_50\best_prior.pt"
    composition = "checkpoints\language_act_composition_balanced_250\best_prior.pt"
    "composition-worst" = "checkpoints\language_act_composition_balanced_250\best_worst_prior.pt"
}
$tasks = @{
    seen_lr = "datasets\aloha2-role-composition\raw_npz\primitive_test\seen_lr"
    unseen_rl = "datasets\aloha2-role-composition\raw_npz\primitive_test\unseen_rl"
    left_tray_push = "datasets\aloha2-role-composition\raw_npz\primitive_test\left_tray_push"
    right_tray_push = "datasets\aloha2-role-composition\raw_npz\primitive_test\right_tray_push"
    left_pick_place = "datasets\aloha2-role-composition\raw_npz\primitive_test\left_pick_place"
    right_pick_place = "datasets\aloha2-role-composition\raw_npz\primitive_test\right_pick_place"
}

$modelNames = if ($Model -eq "all") { @("seen", "composition", "composition-worst") } else { @($Model) }
$taskNames = switch ($Suite) {
    "main" { @("seen_lr", "unseen_rl") }
    "primitives" { @("left_tray_push", "right_tray_push", "left_pick_place", "right_pick_place") }
    default { @("seen_lr", "unseen_rl", "left_tray_push", "right_tray_push", "left_pick_place", "right_pick_place") }
}

foreach ($modelName in $modelNames) {
    $checkpoint = $models[$modelName]
    if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
    foreach ($taskName in $taskNames) {
        $episodeDir = $tasks[$taskName]
        if (-not (Test-Path $episodeDir)) { throw "Missing test set: $episodeDir" }
        $output = "results\act_balanced\$modelName\$taskName"
        Write-Host "`n=== model=$modelName task=$taskName ===" -ForegroundColor Cyan
        & $Python -u evaluation\evaluate_language_act_suite.py `
            --checkpoint $checkpoint `
            --episode-dir $episodeDir `
            --output $output `
            --execute-actions $ExecuteActions `
            --max-actions $MaxActions `
            --device $Device
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: model=$modelName task=$taskName" }
    }
}

Write-Host "`nAll requested ACT evaluations completed." -ForegroundColor Green

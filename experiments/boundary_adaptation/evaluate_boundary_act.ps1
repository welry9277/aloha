param(
    [ValidateSet("all", "main", "primitives", "boundary", "hybrid")]
    [string]$Suite = "all",
    [ValidateSet("best-prior", "best-worst", "latest")]
    [string]$Checkpoint = "best-worst",
    [string]$Device = "cuda",
    [int]$ExecuteActions = 10,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

$CheckpointFiles = @{
    "best-prior" = "best_prior.pt"
    "best-worst" = "best_worst_prior.pt"
    "latest" = "latest.pt"
}
$CheckpointPath = Join-Path $ProjectRoot (
    "checkpoints\language_act_boundary_replacement_250\" +
    $CheckpointFiles[$Checkpoint]
)
if (-not (Test-Path $CheckpointPath)) {
    throw "Missing checkpoint: $CheckpointPath"
}

$DataRoot = Join-Path $ProjectRoot "datasets\aloha2-role-composition\raw_npz"
$ResultRoot = Join-Path $ProjectRoot (
    "results\act_boundary_replacement\" + $Checkpoint.Replace("-", "_")
)
$LimitArgs = @()
if ($Limit -gt 0) {
    $LimitArgs = @("--limit", "$Limit")
}

$TaskDirectories = [ordered]@{
    "seen_lr" = Join-Path $DataRoot "primitive_test\seen_lr"
    "unseen_rl" = Join-Path $DataRoot "primitive_test\unseen_rl"
    "left_tray_push" = Join-Path $DataRoot "primitive_test\left_tray_push"
    "right_tray_push" = Join-Path $DataRoot "primitive_test\right_tray_push"
    "left_pick_place" = Join-Path $DataRoot "primitive_test\left_pick_place"
    "right_pick_place" = Join-Path $DataRoot "primitive_test\right_pick_place"
    "left_pick_place_after_right_push" = Join-Path $DataRoot "primitive_test\left_pick_place_after_right_push"
}

switch ($Suite) {
    "main" { $Tasks = @("seen_lr", "unseen_rl") }
    "primitives" {
        $Tasks = @(
            "left_tray_push",
            "right_tray_push",
            "left_pick_place",
            "right_pick_place"
        )
    }
    "boundary" { $Tasks = @("left_pick_place_after_right_push") }
    "hybrid" { $Tasks = @() }
    default { $Tasks = @($TaskDirectories.Keys) }
}

Push-Location $ProjectRoot
try {
    foreach ($Task in $Tasks) {
        $EpisodeDirectory = $TaskDirectories[$Task]
        if (-not (Test-Path $EpisodeDirectory)) {
            throw "Missing test directory: $EpisodeDirectory"
        }
        $Output = Join-Path $ResultRoot $Task
        Write-Host ""
        Write-Host "=== boundary ACT checkpoint=$Checkpoint task=$Task ==="
        & $Python -u evaluation\evaluate_language_act_suite.py `
            --checkpoint $CheckpointPath `
            --episode-dir $EpisodeDirectory `
            --output $Output `
            --execute-actions $ExecuteActions `
            --max-actions 0 `
            --device $Device `
            @LimitArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for task=$Task"
        }
    }

    if ($Suite -in @("all", "hybrid")) {
        $UnseenDirectory = $TaskDirectories["unseen_rl"]
        foreach ($Mode in @("expert-expert", "expert-act", "act-expert")) {
            $Output = Join-Path $ResultRoot "hybrid\$Mode"
            Write-Host ""
            Write-Host "=== boundary ACT checkpoint=$Checkpoint hybrid=$Mode ==="
            & $Python -u evaluation\evaluate_hybrid_transition.py `
                --mode $Mode `
                --checkpoint $CheckpointPath `
                --episode-dir $UnseenDirectory `
                --output $Output `
                --execute-actions $ExecuteActions `
                --push-max-actions 70 `
                --pnp-max-actions 140 `
                --device $Device `
                @LimitArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Hybrid evaluation failed for mode=$Mode"
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Boundary ACT evaluation completed: $ResultRoot"

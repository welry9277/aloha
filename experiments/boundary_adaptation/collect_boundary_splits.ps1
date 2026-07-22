param(
    [int]$TrainEpisodes = 50,
    [int]$ValEpisodes = 10,
    [int]$TestEpisodes = 20,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

$DataRoot = Join-Path $ProjectRoot "datasets\aloha2-role-composition\raw_npz"
$Collector = Join-Path $PSScriptRoot "collect_boundary_dataset.py"
$ResumeArg = @()
if ($Resume) {
    $ResumeArg = @("--resume")
}

$Splits = @(
    @{
        Name = "train"
        Episodes = $TrainEpisodes
        Seed = 40000
        Attempts = [Math]::Max($TrainEpisodes * 3, $TrainEpisodes)
    },
    @{
        Name = "val"
        Episodes = $ValEpisodes
        Seed = 50000
        Attempts = [Math]::Max($ValEpisodes * 3, $ValEpisodes)
    },
    @{
        Name = "primitive_test"
        Episodes = $TestEpisodes
        Seed = 60000
        Attempts = [Math]::Max($TestEpisodes * 3, $TestEpisodes)
    }
)

Push-Location $ProjectRoot
try {
    foreach ($Split in $Splits) {
        $Output = Join-Path $DataRoot (
            "$($Split.Name)\left_pick_place_after_right_push"
        )
        Write-Host ""
        Write-Host "=== collecting boundary $($Split.Name): $($Split.Episodes) ==="
        & $Python -u $Collector `
            --episodes $Split.Episodes `
            --output $Output `
            --seed $Split.Seed `
            --max-attempts $Split.Attempts `
            @ResumeArg
        if ($LASTEXITCODE -ne 0) {
            throw "Boundary collection failed for split=$($Split.Name)"
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Boundary train/val/test collection completed."

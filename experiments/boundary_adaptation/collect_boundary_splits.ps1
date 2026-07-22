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

$Tasks = @(
    @{
        Name = "left_pick_place_after_right_push"
        SeedOffset = 0
    },
    @{
        Name = "right_pick_place_after_left_push"
        SeedOffset = 30000
    }
)

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
    foreach ($Task in $Tasks) {
        foreach ($Split in $Splits) {
            $Output = Join-Path $DataRoot (
                "$($Split.Name)\$($Task.Name)"
            )
            $TaskSeed = $Split.Seed + $Task.SeedOffset
            Write-Host ""
            Write-Host (
                "=== collecting $($Task.Name) $($Split.Name): " +
                "$($Split.Episodes) ==="
            )
            & $Python -u $Collector `
                --task $Task.Name `
                --episodes $Split.Episodes `
                --output $Output `
                --seed $TaskSeed `
                --max-attempts $Split.Attempts `
                @ResumeArg
            if ($LASTEXITCODE -ne 0) {
                throw (
                    "Boundary collection failed for task=$($Task.Name), " +
                    "split=$($Split.Name)"
                )
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Symmetric boundary train/val/test collection completed."

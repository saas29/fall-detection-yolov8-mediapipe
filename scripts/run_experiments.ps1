# Reproduce the paper's experiments on the laptop.
#   powershell -ExecutionPolicy Bypass -File scripts/run_experiments.ps1
# URFD: download (if needed) -> labels -> eval angle + aspect -> benchmark -> report.
# Le2i / custom clips are evaluated too if their label files exist in data/.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".venv\Scripts\python.exe"

if (-not (Test-Path "videos\urfd\urfall-cam0-falls.csv")) {
    & $py tools/download_urfd.py
}
& $py tools/make_labels.py urfd --root videos/urfd --out data/urfd

foreach ($m in @("angle", "aspect")) {
    & $py fall_detection.py eval --dataset URFD --method $m --videos videos/urfd --crop 320,0,320,240 `
        --labels data/urfd_labels.csv --frame-labels data/urfd_frame_labels.csv
}

if (Test-Path "data\le2i_labels.csv") {
    foreach ($m in @("angle", "aspect")) {
        & $py fall_detection.py eval --dataset Le2i --method $m --videos videos/le2i `
            --labels data/le2i_labels.csv --frame-labels data/le2i_frame_labels.csv
    }
}

if (Test-Path "data\custom_labels.csv") {
    foreach ($m in @("angle", "aspect")) {
        & $py fall_detection.py eval --dataset Live --method $m --videos videos/custom `
            --labels data/custom_labels.csv
    }
}

& $py fall_detection.py benchmark --source videos/urfd/fall-01-cam0.mp4 --crop 320,0,320,240 --name benchmark_laptop_angle
& $py fall_detection.py benchmark --source videos/urfd/fall-01-cam0.mp4 --crop 320,0,320,240 --method aspect --name benchmark_laptop_aspect
& $py fall_detection.py report

Write-Host "`nDone - open results/report.md"

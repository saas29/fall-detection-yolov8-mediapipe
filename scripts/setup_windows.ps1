# One-time laptop setup (Windows, CPU only).
#   powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
# Creates .venv with Python 3.11, installs CPU PyTorch + requirements,
# runs the unit tests and checks that all models load.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "== creating .venv (Python 3.11) =="
    py -3.11 -m venv .venv
    if (-not $?) { throw "Python 3.11 not found. Install it from python.org (mediapipe 0.10.14 needs <= 3.11)." }
}

$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "== installing PyTorch (CPU) =="
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "== installing requirements =="
& $py -m pip install -r requirements.txt

Write-Host "== unit tests =="
& $py -m unittest discover tests

Write-Host "== checking models load =="
& $py -c "from ultralytics import YOLO; import mediapipe as mp; YOLO('models/yolov8n.pt'); mp.solutions.pose.Pose().close(); print('YOLOv8n + MediaPipe Pose OK')"

Write-Host ""
Write-Host "Setup done. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv"

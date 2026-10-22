#!/usr/bin/env bash
# LATER: Raspberry Pi 5 setup (64-bit Raspberry Pi OS).
#   bash scripts/setup_pi.sh
# Same code as the laptop - only the install differs. mediapipe 0.10.14 needs
# Python <= 3.11 (Bookworm ships 3.11).
set -e
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install gpiozero lgpio   # buzzer alert (--buzzer-pin)

python -m unittest discover tests
echo
echo "Run on the Pi (faster settings):"
echo "  python fall_detection.py demo --source 0 --imgsz 320 --pose-complexity 0 --buzzer-pin 17"
echo "  python fall_detection.py benchmark --source videos/urfd/fall-01-cam0.mp4 --imgsz 320 --pose-complexity 0 --name benchmark_pi5 --power <W>"

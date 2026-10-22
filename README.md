# Fall Detection Using YOLOv8 and MediaPipe

Code for the paper *"Fall Detection Using YOLOv8 and MediaPipe: A Software-Based
Pose Estimation Framework for Real-Time Elderly Monitoring"* (Amrita School of
AI, Coimbatore).

Pipeline (paper Fig. 1): frame capture → YOLOv8n person detection → MediaPipe
Pose on the person ROI → torso tilt angle (Eq. 1) → persistence (N frames) and
latch (M frames) → alert. A bounding-box aspect-ratio rule is included as the
baseline (Sec. III-E, Table IV).

```
θ = atan( |x_hip − x_shoulder| / (|y_hip − y_shoulder| + ε) )      θ ≥ 45° → fall
```

## Project layout

```
fall_detection.py        CLI: demo | eval | benchmark | report
falldet/
  geometry.py            Eq. 1 torso angle, aspect ratio, Table II angle bins
  state.py               persistence (N) + latch (M) state machine   (Sec. III-G)
  detector.py            YOLOv8 + ByteTrack + MediaPipe Pose, per-person state
  alerts.py              console, snapshot, Telegram, beep, Raspberry Pi buzzer
  video_io.py            webcam / video / image-folder input, fixed-width resize
  evaluate.py            video-level, frame-level and per-view metrics
  benchmark.py           FPS, latency, memory, CPU             (Table VII)
  report.py              results/report.md + figures (Tables II-IV, VII, Fig. 6)
tools/
  download_urfd.py       downloads UR Fall Detection (cam0 videos + annotations)
  make_labels.py         builds label CSVs for URFD, Le2i and your own clips
tests/test_core.py       unit tests (no models needed)
data/                    generated label CSVs
videos/                  datasets (git-ignored)
results/                 eval/benchmark JSON, report.md, figures
```

## Working in VS Code (laptop)

1. **Open** `fall_detection.code-workspace` (one folder up), or open this
   folder directly with *File → Open Folder*. Install the recommended Python
   extensions when VS Code asks.
2. **Setup once:** *Terminal → Run Task… → "Setup: create .venv + install
   everything"* (or run `scripts\setup_windows.ps1`). Then
   *Ctrl+Shift+P → Python: Select Interpreter → .venv*.
3. **Run:** open *Run and Debug* (Ctrl+Shift+D), pick a configuration, press **F5**:

| Config | What it does |
|---|---|
| 1. Demo: webcam (torso angle) | live detection + beep alert |
| 2. Demo: webcam (aspect ratio) | baseline method live |
| 3 / 3b. Demo: video / URFD clip | asks a path + method |
| 4. Demo → save | writes `results/out.mp4` |
| 5. Eval: URFD quick test | 5 clips with the video shown |
| 6 / 7. Eval: URFD angle / aspect | Table III / Table IV |
| 8 / 9. Eval: Le2i / Live | other datasets |
| 10. Benchmark | Table VII numbers for the laptop |
| 11. Report | builds `results/report.md` + figures |

Breakpoints work in all of them (e.g. in `FallDetector.process`).
**Tasks** (*Terminal → Run Task…*) cover setup, dataset download, label
generation, unit tests and "run all experiments".
**Tests** also appear in the Testing panel (flask icon).

The same code runs on the Raspberry Pi later — only the install
(`scripts/setup_pi.sh`) and a few speed flags differ.

## Setup (manual)

Python **3.9–3.11** is required (mediapipe 0.10.14 has no wheels for 3.12+).

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m unittest discover tests
```

YOLO weights live in `models/yolov8n.pt` (downloaded automatically if missing).

> **OneDrive:** this project sits inside OneDrive, which keeps syncing the
> thousands of files in `.venv/` and `videos/` and eats CPU (slower FPS).
> Pause OneDrive while experimenting, or move the project to e.g.
> `C:\dev\fall_detection_project`.

## 1. Live demo

```
python fall_detection.py demo --source 0                       # webcam
python fall_detection.py demo --source videos\clip.mp4 --save out.mp4
python fall_detection.py demo --source 0 --method aspect       # baseline
python fall_detection.py demo --source 0 --save-raw videos\custom\fall\side\c01.mp4   # record custom data
python fall_detection.py demo --source 0 --headless --buzzer-pin 17   # Raspberry Pi
```

Shows boxes, track IDs, skeleton, torso angle, FPS and a FALL banner. On a
confirmed fall it prints an alert and saves a snapshot to `alerts/`.

**Telegram alerts:** create a bot with @BotFather, then before running:
```
set TELEGRAM_BOT_TOKEN=123456:ABC...
set TELEGRAM_CHAT_ID=987654321
```
Alerts are then sent as a photo message (cool-down `--cooldown`, default 5 s).

## 2. Datasets

**UR Fall Detection (URFD)** – downloaded automatically:
```
python tools/download_urfd.py                  # 30 falls + 40 ADL -> videos/urfd
python tools/make_labels.py urfd --root videos/urfd --out data/urfd
```
URFD mp4s are 640×240 with **depth on the left, RGB on the right** — always
pass `--crop 320,0,320,240` for URFD (the VS Code configs already do).

**Le2i** – download manually (e.g. from Kaggle "Le2i fall detection dataset"),
extract into `videos/le2i/` keeping the `<scene>/Videos` and
`<scene>/Annotation_files` folders, then:
```
python tools/make_labels.py le2i --root videos/le2i --out data/le2i
```

**Custom / live** – record clips with `demo --save-raw` into
`videos/custom/fall/<view>/` and `videos/custom/adl/<view>/`
(`<view>` = side, front, back, front_cross, back_cross for Fig. 6), then:
```
python tools/make_labels.py custom --root videos/custom --out data/custom
```

## 3. Evaluation (Tables III, IV and Fig. 6)

```
python fall_detection.py eval --dataset URFD --videos videos/urfd --labels data/urfd_labels.csv --frame-labels data/urfd_frame_labels.csv
python fall_detection.py eval --dataset URFD --videos videos/urfd --labels data/urfd_labels.csv --frame-labels data/urfd_frame_labels.csv --method aspect
```
Repeat for `Le2i` and `Live`. Each run prints per-clip predictions and saves
`results/<dataset>_<method>.json` + a per-frame log.

- **Video-level**: a clip is predicted "fall" if a fall was confirmed in any frame.
- **Frame-level**: per-frame FALL state vs frame annotations (when given).
- **Per view**: if `labels.csv` has a `view` column.

## 4. Speed (Table VII)

```
python fall_detection.py benchmark --source videos/urfd/fall-01-cam0.mp4
python fall_detection.py benchmark --source 0 --power 3.8    # on the Pi, W from a USB meter
```

## 5. Report

```
python fall_detection.py report
```
Writes `results/report.md` with Table II (angle bins vs triggers), Table III,
Table IV, Fig. 6 (per-view accuracy), Table VII and the list of misclassified
clips, plus PNG figures in `results/figures/`.

## Tuning

| Flag | Meaning | Default |
|---|---|---|
| `--angle` | torso tilt threshold (deg) | 45 |
| `--persist` | N: consecutive fall frames to confirm | 10 |
| `--latch` | M: frames FALL stays on | 30 |
| `--ar` | aspect-ratio threshold (baseline) | 1.0 |
| `--width` | frame width after resize (Sec. III-B) | 640 |
| `--imgsz` | YOLO inference size | 640 |
| `--pose-complexity` | MediaPipe model 0/1/2 (0 = fastest, for the Pi) | 1 |
| `--no-track` | one global fall state instead of per person | off |

## Speed-up options

- `--imgsz 320` — smaller YOLO input (≈2× faster, slightly lower recall).
- `--pose-complexity 0` — lighter MediaPipe model.
- Exported models: `yolo export model=models/yolov8n.pt format=openvino`
  (Intel laptops) or `format=ncnn` (Raspberry Pi), then
  `--weights models/yolov8n_openvino_model`.

## Implementation notes (beyond the paper text)

- **Lying bodies:** MediaPipe's person detector often fails on a horizontal
  body. If pose fails on a wide box (W/H ≥ 0.7), the ROI is enlarged and
  retried upright and rotated ±90°, and landmarks are mapped back before Eq. 1.
- **Unknown frames:** frames where the person isn't detected or has no
  landmarks *hold* the persistence counter instead of resetting it, so short
  detection gaps don't cancel a fall.
- **Per-person state:** ByteTrack IDs give each person their own N/M counters.

## Known limitations

- Top-down cameras make the torso angle unreliable (assumes side / front views).
- Low light and heavy occlusion reduce landmark visibility (no angle → no fall flag).
- Power (W) must be measured externally.

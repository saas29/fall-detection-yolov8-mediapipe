"""
Computational efficiency (Sec. IV-D, Table VII): FPS, per-stage latency,
process memory and CPU load on the current machine (laptop or Raspberry Pi 5).

Power can't be read from software on most boards: measure it with a USB
power meter while this runs and pass it with --power to store it alongside.
"""

import json
import os
import platform
import socket
import statistics
import time
from datetime import datetime

from .detector import DetectorConfig, FallDetector
from .video_io import FrameSource


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run_benchmark(cfg: DetectorConfig, source, frames=300, warmup=15, frame_width=640,
                  crop=None, power_w=None, out_dir="results", name=None):
    import psutil

    proc = psutil.Process(os.getpid())
    detector = FallDetector(cfg)
    src = FrameSource(source, width=frame_width, crop=crop)
    if not src.is_opened():
        print(f"Could not open source: {source}")
        return None

    det_ms, pose_ms, total_ms = [], [], []
    rss_peak = 0
    proc.cpu_percent(None)
    n = 0
    t_start = None

    print(f"benchmarking {frames} frames (+{warmup} warm-up) on {source} ...")
    for frame in src:
        fr = detector.process(frame)
        n += 1
        if n == warmup:
            t_start = time.perf_counter()
            proc.cpu_percent(None)
        if n > warmup:
            det_ms.append(fr.timings["detect_ms"])
            pose_ms.append(fr.timings["pose_ms"])
            total_ms.append(fr.timings["total_ms"])
            rss_peak = max(rss_peak, proc.memory_info().rss)
        if n >= warmup + frames:
            break
    elapsed = (time.perf_counter() - t_start) if t_start else 0.0
    cpu = proc.cpu_percent(None)
    src.release()
    detector.close()

    measured = len(total_ms)
    if measured == 0:
        print("Not enough frames for a benchmark (source too short?).")
        return None

    result = {
        "name": name or f"benchmark_{socket.gethostname()}_{cfg.method}",
        "method": cfg.method,
        "created": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "frames": measured,
        "frame_width": frame_width,
        "fps": measured / elapsed if elapsed else None,
        "detect_ms_mean": statistics.mean(det_ms),
        "pose_ms_mean": statistics.mean(pose_ms),
        "total_ms_mean": statistics.mean(total_ms),
        "total_ms_p95": _pct(total_ms, 0.95),
        "memory_mb_peak": rss_peak / (1024 * 1024),
        "cpu_percent": cpu,
        "power_w": power_w,
        "config": cfg.__dict__,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{result['name']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n--- benchmark ({result['host']}, {cfg.method}) ---")
    print(f"FPS            : {result['fps']:.1f}")
    print(f"latency (mean) : detect {result['detect_ms_mean']:.1f} ms | "
          f"pose+rule {result['pose_ms_mean']:.1f} ms | total {result['total_ms_mean']:.1f} ms")
    print(f"latency p95    : {result['total_ms_p95']:.1f} ms")
    print(f"peak memory    : {result['memory_mb_peak']:.0f} MB")
    print(f"CPU            : {cpu:.0f}% (of one core; >100% = multi-core)")
    print(f"power          : {power_w if power_w is not None else 'not given (use --power W)'}")
    print(f"saved {path}")
    return result

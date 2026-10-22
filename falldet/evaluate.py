"""
Offline evaluation (Sec. IV).

labels.csv        : video,label[,view]       label 1 = fall clip, 0 = ADL clip
                    `video` is a file or an image-frame folder, relative to --videos
                    `view` (optional) e.g. side / front / back / front_cross / back_cross
frame_labels.csv  : video,frame,label        frame is 1-based; label 1 = fall,
                    0 = no fall, -1 = ignore (e.g. URFD transitional frames)

Outputs (in --out, default results/):
  <name>.json         config, per-video predictions, video/frame/per-view metrics
  <name>_frames.csv   one row per detected person per frame (for Table II)
"""

import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime

from .detector import DetectorConfig, FallDetector
from .metrics import compute_metrics, format_metrics
from .video_io import FrameSource


def load_labels(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("video"):
                continue
            rows.append({
                "video": r["video"].strip(),
                "label": int(r["label"]),
                "view": (r.get("view") or "").strip() or None,
            })
    return rows


def load_frame_labels(path):
    """-> {video: {frame_idx: label}}"""
    out = defaultdict(dict)
    if not path:
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["video"].strip()][int(r["frame"])] = int(r["label"])
    return out


def _fmt(v):
    return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else v)


def run_eval(cfg: DetectorConfig, videos_dir, labels_path, frame_labels_path=None,
             dataset="dataset", name=None, out_dir="results", frame_width=640,
             crop=None, limit=None, show=False):
    labels = load_labels(labels_path)
    if limit:
        labels = labels[:limit]
    if not labels:
        print(f"No rows in {labels_path}")
        return None
    frame_labels = load_frame_labels(frame_labels_path)

    name = name or f"{dataset}_{cfg.method}"
    os.makedirs(out_dir, exist_ok=True)
    frames_csv = os.path.join(out_dir, f"{name}_frames.csv")

    detector = FallDetector(cfg)
    per_video = []
    f_true, f_pred = [], []
    total_frames, total_time = 0, 0.0

    print(f"[{name}] evaluating {len(labels)} clips, method={cfg.method}")
    print(f"{'video':40s} {'label':>5s} {'pred':>5s} {'frames':>7s} {'fps':>6s}")

    with open(frames_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["video", "frame", "track_id", "angle", "ar",
                         "flagged", "in_fall", "frame_fall", "gt"])

        for item in labels:
            vid = item["video"]
            path = os.path.join(videos_dir, vid)
            if not os.path.exists(path):
                print(f"  (missing) {vid}")
                continue

            src = FrameSource(path, width=frame_width, crop=crop)
            if not src.is_opened():
                print(f"  (cannot open) {vid}")
                continue

            detector.reset()
            gt_frames = frame_labels.get(vid, {})
            n, first_fall, t_start = 0, None, time.perf_counter()

            for frame in src:
                n += 1
                fr = detector.process(frame)
                if fr.fall and first_fall is None:
                    first_fall = n

                gt = gt_frames.get(n)
                if gt is not None and gt >= 0:
                    f_true.append(gt)
                    f_pred.append(1 if fr.fall else 0)

                if fr.persons:
                    for p in fr.persons:
                        writer.writerow([vid, n, p.track_id, _fmt(p.angle), _fmt(p.ar),
                                         "" if p.flagged is None else int(p.flagged),
                                         int(p.in_fall), int(fr.fall),
                                         "" if gt is None else gt])
                else:
                    writer.writerow([vid, n, "", "", "", 0, 0, int(fr.fall),
                                     "" if gt is None else gt])

                if show:
                    import cv2
                    from .detector import annotate
                    cv2.imshow("eval", annotate(frame, fr, cfg.method))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        show = False
                        cv2.destroyAllWindows()
            src.release()

            dt = time.perf_counter() - t_start
            total_frames += n
            total_time += dt
            pred = 1 if detector.confirmed_this_run else 0

            gt_start = min((k for k, v in gt_frames.items() if v == 1), default=None)
            delay = (first_fall - gt_start) if (first_fall and gt_start) else None

            per_video.append({
                "video": vid, "label": item["label"], "pred": pred, "view": item["view"],
                "frames": n, "first_fall_frame": first_fall,
                "gt_fall_start": gt_start, "delay_frames": delay,
            })
            fps = n / dt if dt > 0 else 0.0
            mark = "" if pred == item["label"] else "  <-- wrong"
            print(f"{vid[:40]:40s} {item['label']:>5d} {pred:>5d} {n:>7d} {fps:>6.1f}{mark}")

    detector.close()
    if not per_video:
        print("No videos evaluated (check --videos path and labels file).")
        return None

    y_true = [v["label"] for v in per_video]
    y_pred = [v["pred"] for v in per_video]
    video_m = compute_metrics(y_true, y_pred)
    frame_m = compute_metrics(f_true, f_pred) if f_true else None

    view_m = {}
    by_view = defaultdict(list)
    for v in per_video:
        if v["view"]:
            by_view[v["view"]].append(v)
    for view, vs in by_view.items():
        view_m[view] = compute_metrics([v["label"] for v in vs], [v["pred"] for v in vs])

    delays = [v["delay_frames"] for v in per_video if v["delay_frames"] is not None]

    result = {
        "name": name,
        "dataset": dataset,
        "method": cfg.method,
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {**cfg.__dict__, "frame_width": frame_width, "crop": crop},
        "videos": per_video,
        "metrics": video_m,
        "frame_metrics": frame_m,
        "view_metrics": view_m,
        "mean_delay_frames": (sum(delays) / len(delays)) if delays else None,
        "processing_fps": total_frames / total_time if total_time else None,
        "frames_csv": os.path.basename(frames_csv),
    }
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n--- {name} ---")
    print(f"video-level : {format_metrics(video_m)}")
    if frame_m:
        print(f"frame-level : {format_metrics(frame_m)}")
    for view, m in sorted(view_m.items()):
        print(f"view {view:12s}: {format_metrics(m)}")
    if result["mean_delay_frames"] is not None:
        print(f"mean detection delay: {result['mean_delay_frames']:.1f} frames")
    print(f"processing FPS: {result['processing_fps']:.1f}")
    print(f"saved {json_path} and {frames_csv}")
    return result

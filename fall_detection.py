"""
fall_detection.py
=================
Fall Detection Using YOLOv8 and MediaPipe - command line entry point.

Modes
-----
    demo       live webcam / video with alerts
    eval       evaluate on a labelled dataset (Tables III / IV, Fig. 6)
    benchmark  FPS / latency / memory on this machine (Table VII)
    report     build results/report.md + figures from saved results

Examples
--------
    python fall_detection.py demo --source 0
    python fall_detection.py demo --source videos\\clip.mp4 --save out.mp4
    python fall_detection.py demo --source 0 --method aspect

    python fall_detection.py eval --dataset URFD --videos videos/urfd \\
        --labels data/urfd_labels.csv --frame-labels data/urfd_frame_labels.csv
    python fall_detection.py eval ... --method aspect

    python fall_detection.py benchmark --source videos/urfd/fall-01-cam0.mp4
    python fall_detection.py report

Press q to quit any video window.
"""

import argparse
import time

from falldet.detector import DetectorConfig


def _parse_crop(s):
    if not s:
        return None
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop needs x,y,w,h")
    return tuple(parts)


def config_from_args(args):
    return DetectorConfig(
        method=args.method,
        angle_thresh=args.angle,
        ar_thresh=args.ar,
        persist_frames=args.persist,
        latch_frames=args.latch,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        pose_complexity=args.pose_complexity,
        track=not args.no_track,
        device=args.device,
    )


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #

def run_demo(args):
    import cv2

    from falldet.alerts import AlertManager
    from falldet.detector import FallDetector, annotate
    from falldet.video_io import FrameSource

    cfg = config_from_args(args)
    detector = FallDetector(cfg)
    alerts = AlertManager(cooldown_s=args.cooldown, snapshot_dir=args.snapshots,
                          beep=args.beep, buzzer_pin=args.buzzer_pin)

    src = FrameSource(args.source, width=args.width, crop=args.crop)
    if not src.is_opened():
        print(f"Could not open source: {args.source}")
        return

    writer = raw_writer = None
    fps_smooth = None
    t_prev = time.perf_counter()
    print("Press q to quit.")

    for frame in src:
        fr = detector.process(frame)

        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_smooth = inst if fps_smooth is None else 0.9 * fps_smooth + 0.1 * inst

        vis = annotate(frame, fr, cfg.method, fps=fps_smooth, draw_skeleton=not args.no_skeleton)
        if fr.newly_confirmed:
            alerts.trigger(vis, "Fall detected")

        if args.save and writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
        if args.save_raw and raw_writer is None:
            h, w = frame.shape[:2]
            raw_writer = cv2.VideoWriter(args.save_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src.fps, (w, h))
        if writer is not None:
            writer.write(vis)
        if raw_writer is not None:
            raw_writer.write(frame)

        if not args.headless:
            cv2.imshow("Fall Detection (YOLOv8 + MediaPipe)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    src.release()
    detector.close()
    for w in (writer, raw_writer):
        if w is not None:
            w.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"alerts sent: {alerts.count}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description="Fall detection: YOLOv8 + MediaPipe Pose")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("detector")
    g.add_argument("--method", choices=["angle", "aspect"], default="angle",
                   help="angle = torso tilt (proposed), aspect = AR baseline")
    g.add_argument("--angle", type=float, default=45.0, help="torso angle threshold, deg (%(default)s)")
    g.add_argument("--ar", type=float, default=1.0, help="aspect-ratio threshold W/H (%(default)s)")
    g.add_argument("--persist", type=int, default=10, help="N: frames to confirm a fall (%(default)s)")
    g.add_argument("--latch", type=int, default=30, help="M: frames FALL is latched (%(default)s)")
    g.add_argument("--weights", default="models/yolov8n.pt", help="YOLOv8 weights (%(default)s)")
    g.add_argument("--conf", type=float, default=0.3, help="YOLO confidence (%(default)s)")
    g.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (%(default)s)")
    g.add_argument("--pose-complexity", type=int, choices=[0, 1, 2], default=1,
                   help="MediaPipe model complexity (%(default)s); 0 is fastest")
    g.add_argument("--no-track", action="store_true",
                   help="disable per-person tracking (single global fall state)")
    g.add_argument("--device", default=None, help="torch device, e.g. cpu or 0")
    g.add_argument("--width", type=int, default=640,
                   help="resize frames to this width before processing, 0 = off (%(default)s)")
    g.add_argument("--crop", type=_parse_crop, default=None,
                   help="crop source frames to x,y,w,h before resizing")

    d = sub.add_parser("demo", parents=[common], help="live webcam / video demo")
    d.add_argument("--source", default="0", help="0 = webcam, or video file / image folder")
    d.add_argument("--save", default=None, help="save annotated video (mp4)")
    d.add_argument("--save-raw", default=None, help="save raw input (to build a custom dataset)")
    d.add_argument("--cooldown", type=float, default=5.0, help="seconds between alerts (%(default)s)")
    d.add_argument("--snapshots", default="alerts", help="folder for alert snapshots")
    d.add_argument("--beep", action="store_true", help="beep on alert (Windows)")
    d.add_argument("--buzzer-pin", type=int, default=None, help="GPIO pin of a buzzer (Raspberry Pi)")
    d.add_argument("--no-skeleton", action="store_true", help="don't draw pose landmarks")
    d.add_argument("--headless", action="store_true", help="no window (e.g. Pi over SSH)")

    e = sub.add_parser("eval", parents=[common], help="evaluate on labelled clips")
    e.add_argument("--videos", required=True, help="root folder of the clips")
    e.add_argument("--labels", required=True, help="CSV: video,label[,view]")
    e.add_argument("--frame-labels", default=None, help="CSV: video,frame,label (optional)")
    e.add_argument("--dataset", default="dataset", help="dataset name used in the report")
    e.add_argument("--name", default=None, help="run name (default <dataset>_<method>)")
    e.add_argument("--out", default="results", help="output folder (%(default)s)")
    e.add_argument("--limit", type=int, default=None, help="only the first N clips (quick test)")
    e.add_argument("--show", action="store_true", help="show frames while evaluating")

    b = sub.add_parser("benchmark", parents=[common], help="FPS / latency / memory")
    b.add_argument("--source", required=True, help="video file, image folder or webcam index")
    b.add_argument("--frames", type=int, default=300, help="frames to measure (%(default)s)")
    b.add_argument("--power", type=float, default=None, help="measured power in W (USB meter)")
    b.add_argument("--name", default=None, help="result name")
    b.add_argument("--out", default="results", help="output folder (%(default)s)")

    r = sub.add_parser("report", help="build results/report.md + figures")
    r.add_argument("--results", default="results", help="results folder (%(default)s)")
    return p


def main():
    args = build_parser().parse_args()

    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "eval":
        from falldet.evaluate import run_eval
        run_eval(config_from_args(args), args.videos, args.labels, args.frame_labels,
                 dataset=args.dataset, name=args.name, out_dir=args.out,
                 frame_width=args.width, crop=args.crop, limit=args.limit, show=args.show)
    elif args.mode == "benchmark":
        from falldet.benchmark import run_benchmark
        run_benchmark(config_from_args(args), args.source, frames=args.frames,
                      frame_width=args.width, crop=args.crop, power_w=args.power,
                      out_dir=args.out, name=args.name)
    elif args.mode == "report":
        from falldet.report import build_report
        build_report(args.results)


if __name__ == "__main__":
    main()

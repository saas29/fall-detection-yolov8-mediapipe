"""
Build labels.csv (video-level) and frame_labels.csv (frame-level) for eval.

URFD   python tools/make_labels.py urfd   --root videos/urfd --out data/urfd
Le2i   python tools/make_labels.py le2i   --root videos/le2i --out data/le2i
Custom python tools/make_labels.py custom --root videos/custom --out data/custom

Writes <out>_labels.csv and <out>_frame_labels.csv (when frame info exists).

URFD   : files fall-XX-cam0.mp4 / adl-XX-cam0.mp4 (or *-cam0-rgb/ image folders).
         urfall-cam0-*.csv label -1 = not lying -> 0, 1 = lying -> 1,
         0 = transitional -> -1 (ignored, as in the dataset's protocol).
Le2i   : <scene>/Videos/video (N).avi with <scene>/Annotation_files/video (N).txt
         whose first two lines are the fall start / end frame (0 0 = no fall).
Custom : <root>/fall/*.mp4 and <root>/adl/*.mp4 (subfolders = view names are
         also picked up: <root>/fall/side/clip.mp4 -> view "side").
"""

import argparse
import csv
import glob
import os
import re

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _write(out_prefix, rows, frame_rows):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    lab = f"{out_prefix}_labels.csv"
    with open(lab, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "label", "view"])
        for r in rows:
            w.writerow([r["video"], r["label"], r.get("view") or ""])
    print(f"wrote {lab}  ({len(rows)} clips: "
          f"{sum(r['label'] for r in rows)} fall / {sum(1 - r['label'] for r in rows)} ADL)")

    if frame_rows:
        fl = f"{out_prefix}_frame_labels.csv"
        with open(fl, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "frame", "label"])
            w.writerows(frame_rows)
        print(f"wrote {fl}  ({len(frame_rows)} frames)")


def _rel(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


# --------------------------------------------------------------------------- #

def urfd(root, out):
    clips = {}
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        m = re.match(r"^((fall|adl)-\d+)-cam0(\.mp4|-rgb)?$", os.path.basename(p))
        if not m:
            continue
        seq = m.group(1)
        # prefer the image-folder version if both exist (pure RGB frames)
        if seq not in clips or os.path.isdir(p):
            clips[seq] = _rel(p, root)

    rows = [{"video": v, "label": 1 if s.startswith("fall") else 0}
            for s, v in sorted(clips.items())]

    frame_rows = []
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print(f"(no {name} in {root}: skipping frame labels from it)")
            continue
        with open(path, newline="") as f:
            for r in csv.reader(f):
                if len(r) < 3 or r[0] not in clips:
                    continue
                try:
                    frame, lab = int(r[1]), int(float(r[2]))
                except ValueError:
                    continue
                mapped = {-1: 0, 0: -1, 1: 1}.get(lab)
                if mapped is not None:
                    frame_rows.append([clips[r[0]], frame, mapped])
    _write(out, rows, frame_rows)


def _read_le2i_annotation(path):
    nums = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if re.fullmatch(r"\d+", line):
                nums.append(int(line))
                if len(nums) == 2:
                    break
            elif nums:
                break
    if len(nums) < 2:
        return None
    return nums[0], nums[1]


def le2i(root, out):
    rows, frame_rows, missing = [], [], 0
    for vpath in sorted(glob.glob(os.path.join(root, "**", "*.avi"), recursive=True)):
        vdir = os.path.dirname(vpath)
        stem = os.path.splitext(os.path.basename(vpath))[0]
        ann = None
        for cand in ("Annotation_files", "Annotations_files", "Annotation"):
            p = os.path.join(os.path.dirname(vdir), cand, stem + ".txt")
            if os.path.exists(p):
                ann = p
                break
        if ann is None:
            missing += 1
            continue
        se = _read_le2i_annotation(ann)
        if se is None:
            missing += 1
            continue
        start, end = se
        is_fall = 1 if (end > 0 and end >= start) else 0
        rel = _rel(vpath, root)
        scene = _rel(vpath, root).split("/")[0]
        rows.append({"video": rel, "label": is_fall, "view": None, "scene": scene})
        if is_fall:
            for fr in range(max(1, start), end + 1):
                frame_rows.append([rel, fr, 1])
            # frames outside the interval are known non-fall only up to video length;
            # mark a margin before the fall as 0 so frame-level FP are counted too
            for fr in range(1, max(1, start)):
                frame_rows.append([rel, fr, 0])
    if missing:
        print(f"({missing} videos without a usable annotation file were skipped)")
    _write(out, rows, frame_rows)


def custom(root, out):
    rows = []
    for cls, label in (("fall", 1), ("adl", 0), ("nonfall", 0), ("no_fall", 0)):
        base = os.path.join(root, cls)
        if not os.path.isdir(base):
            continue
        for p in sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True)):
            if not p.lower().endswith(VIDEO_EXTS):
                continue
            sub = os.path.relpath(os.path.dirname(p), base)
            view = None if sub == "." else sub.replace("\\", "/").split("/")[0]
            rows.append({"video": _rel(p, root), "label": label, "view": view})
    if not rows:
        print(f"no clips found under {root}/fall or {root}/adl")
        return
    _write(out, rows, [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", choices=["urfd", "le2i", "custom"])
    ap.add_argument("--root", required=True, help="dataset folder")
    ap.add_argument("--out", required=True, help="output prefix, e.g. data/urfd")
    args = ap.parse_args()
    {"urfd": urfd, "le2i": le2i, "custom": custom}[args.dataset](args.root, args.out)


if __name__ == "__main__":
    main()

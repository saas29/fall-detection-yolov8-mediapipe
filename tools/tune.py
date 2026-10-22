"""
Offline threshold tuning from an eval run's per-frame log - no models needed,
a full grid takes seconds instead of re-running YOLO + MediaPipe.

    python tools/tune.py results/URFD_angle_frames.csv --labels data/urfd_labels.csv

Re-simulates the persistence / latch decision for every
(angle threshold, N, rule) on the logged per-person angles and aspect ratios:

  rule "angle"   : flagged = angle >= thr                     (paper, Eq. 1)
  rule "hybrid"  : angle rule; if no landmarks, fall back to AR > 1
  rule "either"  : angle >= thr OR AR > ar_thr

Reports the best setting on all clips AND a k-fold cross-validated score
(parameters chosen on k-1 folds, scored on the held-out fold) so the number
you report isn't just fitted to the test clips.
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falldet.evaluate import load_labels  # noqa: E402
from falldet.metrics import compute_metrics  # noqa: E402
from falldet.state import FallStateMachine  # noqa: E402


def load_frames(path):
    """-> {video: [(frame, [(track_id, angle|None, ar|None), ...]), ...]}"""
    by_vid = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            persons = by_vid[r["video"]][int(r["frame"])]
            if r["track_id"] == "":
                continue
            persons.append((r["track_id"],
                            float(r["angle"]) if r["angle"] else None,
                            float(r["ar"]) if r["ar"] else None))
    return {v: sorted(frames.items()) for v, frames in by_vid.items()}


def flag(angle, ar, thr, rule, ar_thr):
    if rule == "angle":
        return None if angle is None else angle >= thr
    if rule == "hybrid":
        if angle is not None:
            return angle >= thr
        return None if ar is None else ar > ar_thr
    if rule == "either":
        a = angle is not None and angle >= thr
        b = ar is not None and ar > ar_thr
        if a or b:
            return True
        return None if (angle is None and ar is None) else False
    raise ValueError(rule)


def simulate(frames, thr, persist, latch, rule, ar_thr, stale=45):
    """True if a fall is confirmed anywhere in the clip (same logic as FallDetector)."""
    tracks, last_seen = {}, {}
    for idx, (_, persons) in enumerate(frames, 1):
        flags = {}
        for tid, angle, ar in persons:
            f = flag(angle, ar, thr, rule, ar_thr)
            prev = flags.get(tid)
            flags[tid] = True if (prev is True or f is True) else (
                False if (prev is False or f is False) else None)
        for tid in flags:
            if tid not in tracks:
                tracks[tid] = FallStateMachine(persist, latch)
            last_seen[tid] = idx
        for tid, sm in list(tracks.items()):
            if tid in flags:
                _, new = sm.update(flags[tid])
                if new:
                    return True
            elif idx - last_seen[tid] > stale:
                del tracks[tid]
            else:
                sm.update(None)
    return False


def score(params, clips, frames, latch):
    thr, n, rule, ar_thr = params
    y_true = [c["label"] for c in clips]
    y_pred = [int(simulate(frames.get(c["video"], []), thr, n, latch, rule, ar_thr)) for c in clips]
    return compute_metrics(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("frames_csv")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--latch", type=int, default=30)
    ap.add_argument("--rules", default="angle,hybrid,either")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--metric", default="f1", choices=["f1", "accuracy"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = load_frames(args.frames_csv)
    clips = [c for c in load_labels(args.labels) if c["video"] in frames]
    if not clips:
        print("no labelled clips found in the frame log")
        return

    grid = [(thr, n, rule, ar_thr)
            for rule in args.rules.split(",")
            for thr in (35, 40, 45, 50, 55, 60)
            for n in (2, 3, 5, 8, 10, 15, 20)
            for ar_thr in ((1.0, 1.2, 1.5) if rule != "angle" else (1.0,))]

    key = lambda m: (m[args.metric], m["accuracy"])  # noqa: E731
    print(f"{len(clips)} clips, {len(grid)} settings\n")

    # --- paper default ---
    base = score((45, 10, "angle", 1.0), clips, frames, args.latch)
    print(f"paper default (45 deg, N=10, angle): acc={base['accuracy']:.3f} "
          f"prec={base['precision']:.3f} rec={base['recall']:.3f} f1={base['f1']:.3f}")

    # --- best per rule on all clips ---
    results = [(p, score(p, clips, frames, args.latch)) for p in grid]
    print("\nbest on ALL clips (optimistic - fitted to the test set):")
    for rule in args.rules.split(","):
        p, m = max(((p, m) for p, m in results if p[2] == rule), key=lambda x: key(x[1]))
        print(f"  {rule:7s} thr={p[0]:>2} N={p[1]:>2} ar>{p[3]:.1f}: acc={m['accuracy']:.3f} "
              f"prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f}")

    # --- k-fold cross-validation per rule ---
    rnd = random.Random(args.seed)
    order = clips[:]
    rnd.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]
    print(f"\n{args.folds}-fold cross-validated (params picked on train folds):")
    for rule in args.rules.split(","):
        rule_grid = [p for p in grid if p[2] == rule]
        y_true, y_pred, picks = [], [], []
        for k in range(args.folds):
            train = [c for i, f in enumerate(folds) if i != k for c in f]
            best = max(rule_grid, key=lambda p: key(score(p, train, frames, args.latch)))
            picks.append(best)
            for c in folds[k]:
                y_true.append(c["label"])
                y_pred.append(int(simulate(frames.get(c["video"], []), best[0], best[1],
                                           args.latch, rule, best[3])))
        m = compute_metrics(y_true, y_pred)
        chosen = ", ".join(f"{p[0]}/{p[1]}" + (f"/{p[3]}" if rule != "angle" else "") for p in picks)
        print(f"  {rule:7s}: acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f}   (thr/N per fold: {chosen})")


if __name__ == "__main__":
    main()

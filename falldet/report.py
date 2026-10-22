"""
Builds the paper's tables/figures from the JSON + CSV files written by
`eval` and `benchmark`:

  Table II   torso angle bin -> avg FALL-state frames per 100 frames
  Table III  per-dataset Acc / Prec / Rec / F1 (angle method)
  Table IV   torso tilt angle vs aspect ratio
  Fig. 6     accuracy per camera view, both methods
  Table VII  FPS / memory / power

Writes results/report.md and PNG figures into results/figures/.
"""

import csv
import glob
import json
import os
from collections import defaultdict

from .geometry import ANGLE_BINS, angle_bin


def _load_json(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def _pct(x):
    return f"{100 * x:.1f}%"


def angle_bin_table(frames_csv):
    """Table II rows from a per-frame log of an angle-method run."""
    counts = [0] * len(ANGLE_BINS)
    triggers = [0] * len(ANGLE_BINS)
    with open(frames_csv, newline="") as f:
        for r in csv.DictReader(f):
            if not r["angle"]:
                continue
            b = angle_bin(float(r["angle"]))
            if b is None:
                continue
            counts[b] += 1
            triggers[b] += int(r["in_fall"])
    rows = []
    for (lo, hi, name), c, t in zip(ANGLE_BINS, counts, triggers):
        rows.append({"range": f"{lo}-{hi}", "posture": name, "frames": c,
                     "per100": (100.0 * t / c) if c else None})
    return rows


def _plot_methods(table4, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(table4)
    metrics = ["accuracy", "recall", "precision", "f1"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        width = 0.38
        xs = range(len(metrics))
        for i, (method, label, color) in enumerate(
                [("angle", "Torso Tilt Angle", "#2a6fdb"), ("aspect", "Aspect Ratio", "#e0703a")]):
            m = table4[ds].get(method)
            if not m:
                continue
            vals = [m[k] for k in metrics]
            ax.bar([x + (i - 0.5) * width for x in xs], vals, width, label=label, color=color)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([k.capitalize() for k in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_views(views, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({v for m in views.values() for v in m})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.38
    for i, (method, label, color) in enumerate(
            [("angle", "Torso Tilt Angle", "#2a6fdb"), ("aspect", "Aspect Ratio", "#e0703a")]):
        vals = [views.get(method, {}).get(n, {}).get("accuracy", 0) for n in names]
        ax.bar([x + (i - 0.5) * width for x in range(len(names))], vals, width,
               label=label, color=color)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by camera view")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_angle_bins(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar([r["range"] for r in rows], [r["per100"] or 0 for r in rows], color="#2a6fdb")
    ax.set_xlabel("Torso angle (deg)")
    ax.set_ylabel("FALL frames per 100 frames")
    ax.set_title("Torso angle vs fall triggers")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(results_dir="results"):
    evals = [r for r in _load_json(os.path.join(results_dir, "*.json"))
             if "metrics" in r and "method" in r]
    benches = [r for r in _load_json(os.path.join(results_dir, "*.json")) if "fps" in r
               and "metrics" not in r]
    if not evals and not benches:
        print(f"No eval/benchmark JSON files in {results_dir}/ - run eval or benchmark first.")
        return None

    # newest run wins per (dataset, method)
    latest = {}
    for r in sorted(evals, key=lambda r: r.get("created", "")):
        latest[(r["dataset"], r["method"])] = r

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lines = ["# Fall Detection - Results", ""]

    # ---- Table III ----
    t3 = [(ds, r) for (ds, m), r in sorted(latest.items()) if m == "angle"]
    if t3:
        lines += ["## Table III - Performance across datasets (Torso Tilt Angle)", "",
                  "| Dataset | Videos | Acc. | Prec. | Rec. | F1 | Frame-level F1 | FPS |",
                  "|---|---|---|---|---|---|---|---|"]
        for ds, r in t3:
            m = r["metrics"]
            fm = r.get("frame_metrics")
            lines.append(f"| {ds} | {m['n']} | {_pct(m['accuracy'])} | {_pct(m['precision'])} | "
                         f"{_pct(m['recall'])} | {_pct(m['f1'])} | "
                         f"{_pct(fm['f1']) if fm else '-'} | {r.get('processing_fps') or 0:.1f} |")
        lines.append("")

    # ---- Table IV ----
    table4 = defaultdict(dict)
    for (ds, method), r in latest.items():
        table4[ds][method] = r["metrics"]
    t4 = {ds: d for ds, d in table4.items() if "angle" in d and "aspect" in d}
    if t4:
        lines += ["## Table IV - Torso Tilt Angle vs Aspect Ratio", "",
                  "| Dataset | Model | Accuracy | Recall | Precision | F1 |",
                  "|---|---|---|---|---|---|"]
        for ds in sorted(t4):
            for method, label in [("angle", "Torso Tilt Angle"), ("aspect", "Aspect Ratio")]:
                m = t4[ds][method]
                lines.append(f"| {ds} | {label} | {_pct(m['accuracy'])} | {_pct(m['recall'])} | "
                             f"{_pct(m['precision'])} | {_pct(m['f1'])} |")
        lines.append("")
        p = os.path.join(fig_dir, "table4_methods.png")
        _plot_methods(t4, p)
        lines += [f"![methods](figures/{os.path.basename(p)})", ""]

    # ---- Fig. 6 ----
    views = defaultdict(dict)
    for (ds, method), r in latest.items():
        for view, m in (r.get("view_metrics") or {}).items():
            views[method][view] = m
    if views:
        names = sorted({v for m in views.values() for v in m})
        lines += ["## Fig. 6 - Accuracy by camera view", "",
                  "| View | Torso Tilt Angle | Aspect Ratio |", "|---|---|---|"]
        for n in names:
            a = views.get("angle", {}).get(n)
            b = views.get("aspect", {}).get(n)
            lines.append(f"| {n} | {_pct(a['accuracy']) if a else '-'} | "
                         f"{_pct(b['accuracy']) if b else '-'} |")
        lines.append("")
        p = os.path.join(fig_dir, "fig6_views.png")
        _plot_views(views, p)
        lines += [f"![views](figures/{os.path.basename(p)})", ""]

    # ---- Table II ----
    bin_counts = None
    for ds, r in t3:
        path = os.path.join(results_dir, r.get("frames_csv", ""))
        if not os.path.isfile(path):
            continue
        rows = angle_bin_table(path)
        if bin_counts is None:
            bin_counts = [{"range": x["range"], "posture": x["posture"], "frames": 0, "trig": 0.0}
                          for x in rows]
        for acc, x in zip(bin_counts, rows):
            acc["frames"] += x["frames"]
            acc["trig"] += (x["per100"] or 0) * x["frames"] / 100.0
    if bin_counts:
        rows = [{**b, "per100": (100.0 * b["trig"] / b["frames"]) if b["frames"] else None}
                for b in bin_counts]
        lines += ["## Table II - Torso angle vs average fall triggers", "",
                  "| Angle (deg) | Posture | Frames | Avg. triggers / 100 frames |",
                  "|---|---|---|---|"]
        for x in rows:
            v = "-" if x["per100"] is None else f"{x['per100']:.0f}"
            lines.append(f"| {x['range']} | {x['posture']} | {x['frames']} | {v} |")
        lines.append("")
        p = os.path.join(fig_dir, "table2_angle_bins.png")
        _plot_angle_bins(rows, p)
        lines += [f"![angle bins](figures/{os.path.basename(p)})", ""]

    # ---- Table VII ----
    if benches:
        lines += ["## Table VII - Computational performance", "",
                  "| Host | Method | FPS | Latency (ms) | Mem (MB) | CPU % | Power (W) |",
                  "|---|---|---|---|---|---|---|"]
        for b in sorted(benches, key=lambda b: b.get("created", "")):
            pw = "-" if b.get("power_w") is None else f"{b['power_w']:.1f}"
            lines.append(f"| {b['host']} ({b.get('machine', '')}) | {b['method']} | "
                         f"{b['fps']:.1f} | {b['total_ms_mean']:.1f} | "
                         f"{b['memory_mb_peak']:.0f} | {b['cpu_percent']:.0f} | {pw} |")
        lines.append("")

    # ---- per-video errors ----
    lines += ["## Misclassified clips", ""]
    any_err = False
    for (ds, method), r in sorted(latest.items()):
        wrong = [v for v in r["videos"] if v["label"] != v["pred"]]
        if wrong:
            any_err = True
            lines.append(f"- **{ds} / {method}**: " + ", ".join(
                f"{v['video']} ({'FN' if v['label'] else 'FP'})" for v in wrong))
    if not any_err:
        lines.append("- none")
    lines.append("")

    out = os.path.join(results_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nsaved {out}")
    return out

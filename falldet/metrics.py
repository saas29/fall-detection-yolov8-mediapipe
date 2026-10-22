"""Accuracy / Precision / Recall / F1 as reported in Tables III-V."""


def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    }


def format_metrics(m):
    return (f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
            f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
            f"(n={m['n']}, tp={m['tp']}, tn={m['tn']}, fp={m['fp']}, fn={m['fn']})")

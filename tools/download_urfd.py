"""
Download the UR Fall Detection dataset (camera 0 videos + frame annotations).

    python tools/download_urfd.py                 # -> videos/urfd/
    python tools/download_urfd.py --out D:/urfd --falls 5 --adls 5   # small subset

Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
(Kwolek & Kepski, "Human fall detection on embedded platform using depth maps
and wireless accelerometer", 2014). 30 falls + 40 ADL sequences, ~1.2 MB each.
"""

import argparse
import os
import sys
import urllib.request

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data/"
CSVS = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]


def fetch(name, out_dir):
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(BASE + name, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dst)
        return "ok"
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"FAILED ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join("videos", "urfd"))
    ap.add_argument("--falls", type=int, default=30, help="number of fall sequences (max 30)")
    ap.add_argument("--adls", type=int, default=40, help="number of ADL sequences (max 40)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = list(CSVS)
    names += [f"fall-{i:02d}-cam0.mp4" for i in range(1, min(args.falls, 30) + 1)]
    names += [f"adl-{i:02d}-cam0.mp4" for i in range(1, min(args.adls, 40) + 1)]

    failed = 0
    for i, n in enumerate(names, 1):
        status = fetch(n, args.out)
        failed += status.startswith("FAILED")
        print(f"[{i:3d}/{len(names)}] {n:28s} {status}")
    print(f"done -> {args.out}" + (f"  ({failed} failed, re-run to retry)" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

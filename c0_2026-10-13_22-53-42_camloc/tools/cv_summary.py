#!/usr/bin/env python3
"""
Score cross-validation results with the official evaluate.py.

Every cv/val_<laps>.csv is scored on each of its held-out laps (GT cut to the lap window).
Prints a per-lap table, the mean and the pooled value over all held-out laps, and the runtime
lines from the fold logs. Saves cv/summary.csv.

    python cv_summary.py --gt gt.csv --laps-csv data/laps.csv --dir cv
"""
import argparse, re, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from eval_table import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--laps-csv", required=True)
    ap.add_argument("--dir", default="cv")
    ap.add_argument("--evaluate", default=str(Path(__file__).with_name("evaluate.py")))
    a = ap.parse_args()

    gt = pd.read_csv(a.gt).sort_values("timestamp_ns")
    laps = pd.read_csv(a.laps_csv)
    t0 = int(gt.timestamp_ns.iloc[0])
    tmp = Path(tempfile.mkdtemp())
    rows, runtime = [], []
    for csv in sorted(Path(a.dir).glob("val_*.csv")):
        fold = csv.stem
        held = [int(x) for x in fold.replace("val_", "").split("_")]
        sub = pd.read_csv(csv).sort_values("timestamp_ns")
        for l in held:
            r = laps[laps.lap == l].iloc[0]
            lo, hi = t0 + int(r.t_start * 1e9), t0 + int(r.t_end * 1e9)
            m = evaluate(a.evaluate, gt, sub, lo, hi, tmp, f"{fold}_{l}")
            m.update(fold=fold, lap=l)
            rows.append(m)
        logf = Path(a.dir) / f"{fold}.log"
        if logf.exists():
            txt = logf.read_text(errors="ignore")
            mm = re.findall(r"per frame: mean ([\d.]+) ms", txt)
            if mm:
                runtime.append((fold, float(mm[-1])))
    if not rows:
        print("no cv/val_*.csv found")
        return

    t = pd.DataFrame(rows)
    cols = ["fold", "lap", "evaluated", "coverage %", "pos RMSE", "median", "p95", "lat RMSE",
            "lon RMSE", "yaw RMSE deg", "<0.25m %", "<0.5m %", "<1m %", "ranked"]
    t = t[cols].sort_values("lap")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nper held-out lap (official evaluate.py):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:.1f}"))

    n = t["evaluated"].values
    print("\nover all held-out laps:")
    for k in ["pos RMSE", "lat RMSE", "lon RMSE", "yaw RMSE deg"]:
        v = t[k].values
        pooled = np.sqrt(np.sum(n * v ** 2) / n.sum())
        print(f"  {k:14s} mean {v.mean():.3f} | pooled {pooled:.3f} | min {v.min():.3f} "
              f"(lap {int(t.lap.values[v.argmin()])}) | max {v.max():.3f} "
              f"(lap {int(t.lap.values[v.argmax()])}) | std {v.std():.3f}")
    for k in ["<0.25m %", "<1m %", "coverage %"]:
        print(f"  {k:14s} mean {t[k].mean():.1f}")
    print(f"  ranked on every lap: {bool(t.ranked.all())}")
    if runtime:
        print("\nruntime (mean per frame, from the fold logs):")
        for f, ms in runtime:
            print(f"  {f}: {ms:.1f} ms")
    out = Path(a.dir) / "summary.csv"
    t.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
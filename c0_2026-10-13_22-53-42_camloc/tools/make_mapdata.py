#!/usr/bin/env python3
"""
Data folder for building a map in which some laps use synthetic day images (e.g. Qwen) instead
of the real night ones. Same layout as data/ (images/<cam>/<stamp>.jpg, frames.parquet with GT,
calib, ...), so colmap_map.py / dino_desc.py work unchanged. Images are symlinks, nothing is
copied.

    python make_mapdata.py --data data --synth data_synth_qwen --synth-laps 3 --out data_mapqwen
    python colmap_map.py build --data data_mapqwen --laps 0 1 2 3 4 5 7 8 \
        --cams fl fr cl cr r rl rr --kf-dist 2 --num-pairs 15 --out maps/colmap_7cam_kf2_qwen

data_synth_qwen comes from synth_data.py (images_synth/<cam>/<stamp>.jpg for the synthetic laps).
"""
import argparse, os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--synth", required=True, help="folder from synth_data.py")
    ap.add_argument("--synth-laps", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, synth, out = Path(a.data).resolve(), Path(a.synth).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(data / "frames.parquet")
    n_syn = n_night = n_missing = 0
    for r in f.itertuples():
        dst = out / "images" / r.cam / f"{r.stamp_ns}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        src = data / r.path
        if r.lap in a.synth_laps:
            s = synth / "images_synth" / r.cam / f"{r.stamp_ns}.jpg"
            if s.exists():
                src, n_syn = s, n_syn + 1
            else:
                n_missing += 1
        else:
            n_night += 1
        os.symlink(src, dst)
    f = f.copy()
    f["path"] = [f"images/{c}/{s}.jpg" for c, s in zip(f.cam, f.stamp_ns)]
    f.to_parquet(out / "frames.parquet")
    for name in ["calib.json", "state.parquet", "laps.csv", "motion_calib.json", "track_bounds.json",
                 "gt_tf.parquet", "masks"]:
        s = data / name
        if s.exists() and not (out / name).exists():
            os.symlink(s, out / name)
    print(f"{out}: {n_syn} synthetic images (laps {a.synth_laps}), {n_night} night images, "
          f"{n_missing} synthetic missing -> night kept")


if __name__ == "__main__":
    main()
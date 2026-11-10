#!/usr/bin/env python3
"""
A "day" copy of the validation laps WITH ground truth, from the synthetic night->day mcaps.

The synthetic mcaps are the training run translated to day (same cameras, same stamps), so the
GT of the training run applies to them. This makes a data folder that is identical to data/
except that the images of the chosen laps are the synthetic day ones. The map keyframes (other
laps) still point to the real night images, exactly as in the night validation.

    python synth_data.py --data data --mcap ../drive_data/cyclegan.mcap --laps 3 6 --out data_synth_cyclegan
    python featmap_mix.py validate --data data_synth_cyclegan --map maps/colmap_7cam_kf2 --laps 3 6 \
        --state data_synth_cyclegan/state.parquet --dino --out synth_cyclegan.csv
    python3 evaluate.py gt_lap3.csv synth_cyclegan.csv      # real metrics, "day" images

Use it to tune day settings (exposure, CLAHE, number of keypoints, ...) with GT, then confirm
on the real day bag (fix rate). Synthetic day is only a proxy for real day.
"""
import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

IMG = re.compile(r"/camera_([^/]+)/(compressed_image|image_raw|image)$")


def stamp(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--laps", nargs="+", type=int, default=[3, 6])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-ms", type=float, default=20.0)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    f = pd.read_parquet(data / "frames.parquet")
    want = f[f.lap.isin(a.laps)]
    need = {c: np.sort(g.stamp_ns.values) for c, g in want.groupby("cam")}
    tol = int(a.tol_ms * 1e6)
    newpath = {}
    for m in read_ros2_messages(a.mcap):
        mi = IMG.search(m.channel.topic)
        if not mi or mi.group(1) not in need:
            continue
        cam, s = mi.group(1), stamp(m.ros_msg.header)
        arr = need[cam]
        j = np.searchsorted(arr, s)
        cand = [x for x in (j - 1, j) if 0 <= x < len(arr) and abs(int(arr[x]) - s) <= tol]
        if not cand:
            continue
        st = int(arr[min(cand, key=lambda x: abs(int(arr[x]) - s))])
        if (cam, st) in newpath:
            continue
        msg = m.ros_msg
        rel = Path("images_synth") / cam / f"{st}.jpg"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        if hasattr(msg, "format"):                         # CompressedImage: keep the bytes
            (out / rel).write_bytes(bytes(msg.data))
        else:
            import cv2
            im = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding.lower().startswith("rgb"):
                im = im[:, :, ::-1]
            cv2.imwrite(str(out / rel), im)
        newpath[(cam, st)] = str(rel)

    # frames: chosen laps -> synthetic images; everything else -> original images (relative path)
    f = f.copy()
    key = list(zip(f.cam, f.stamp_ns))
    repl = np.array([newpath.get(k) for k in key], dtype=object)
    has = pd.notna(repl)
    f["path"] = [r if h else os.path.relpath(data / p, out) for r, h, p in zip(repl, has, f.path)]
    missing = f.lap.isin(a.laps) & ~has
    f.to_parquet(out / "frames.parquet")
    for name in ["calib.json", "state.parquet", "laps.csv", "motion_calib.json", "track_bounds.json",
                 "gt_tf.parquet", "masks"]:
        src = data / name
        if src.exists() and not (out / name).exists():
            os.symlink(os.path.relpath(src, out), out / name)
    tot = int(f.lap.isin(a.laps).sum())
    print(f"synthetic images for laps {a.laps}: {tot - int(missing.sum())}/{tot} "
          f"({int(missing.sum())} missing -> real night image kept) -> {out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Add DINOv2 descriptors of synthetic day versions of the map keyframes.

The synthetic mcaps (e.g. CycleGAN / Marigold night->day of the training run) contain the same
camera topics and header stamps as the training run. For every map keyframe the matching
synthetic image is found by (camera, stamp), its DINOv2 descriptor is computed, and a new map
folder is made where <cam>_dino.npy = normalised mean of the night descriptor and the day ones.
Everything else in the new map folder is a symlink to the original map, so the localizer is
used unchanged: only keyframe SELECTION changes (which keyframes are matched), not the 3D points.
The per-source descriptors are saved too (<cam>_dino_<source>.npy) for later experiments.

    python dino_aug.py --data data --map maps/colmap_7cam_kf2 \
        --mcap ../drive_data/cyclegan.mcap ../drive_data/marigold_v2.mcap \
        --out maps/colmap_7cam_kf2_dayaug
"""
import argparse, json, os, re, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

from dino_desc import Dino

IMG = re.compile(r"/camera_([^/]+)/(compressed_image|image_raw|image)$")


def stamp(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def decode(msg):
    if hasattr(msg, "format") or msg.__class__.__name__.lower().startswith("compressed"):
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    h, w = msg.height, msg.width
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(h, w, -1)
    return a[:, :, ::-1].copy() if msg.encoding.lower().startswith("rgb") else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--map", default="maps/colmap_7cam_kf2")
    ap.add_argument("--mcap", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-ms", type=float, default=20.0, help="stamp tolerance for matching")
    ap.add_argument("--night-weight", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    mp, out = Path(a.map), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = json.loads((mp / "meta.json").read_text())["cams"]
    index = pd.read_parquet(mp / "index.parquet")
    need = {c: np.sort(index[index.cam == c].stamp_ns.values) for c in cams}
    pos = {c: {s: i for i, s in enumerate(index[index.cam == c].stamp_ns.values)} for c in cams}
    dino = Dino()
    tol = int(a.tol_ms * 1e6)

    day = {c: [] for c in cams}                          # list of (source, kf index array, desc)
    for path in a.mcap:
        src = Path(path).stem
        t0 = time.time()
        buf = defaultdict(list)                          # cam -> [(kf idx, image)]
        got = {c: np.full(len(need[c]), -1) for c in cams}
        desc = {c: np.zeros((len(need[c]), 0), np.float32) for c in cams}
        acc = {c: {} for c in cams}

        def flush(cam):
            if not buf[cam]:
                return
            idx, ims = zip(*buf[cam])
            D = dino(list(ims))
            for i, d in zip(idx, D):
                acc[cam][i] = d
            buf[cam].clear()

        n_msgs = 0
        for m in read_ros2_messages(path):
            mi = IMG.search(m.channel.topic)
            if not mi or mi.group(1) not in need:
                continue
            cam = mi.group(1)
            s = stamp(m.ros_msg.header)
            arr = need[cam]
            j = np.searchsorted(arr, s)
            best = None
            for jj in (j - 1, j):
                if 0 <= jj < len(arr) and abs(int(arr[jj]) - s) <= tol:
                    if best is None or abs(int(arr[jj]) - s) < abs(int(arr[best]) - s):
                        best = jj
            if best is None:
                continue
            kf = pos[cam][int(arr[best])]
            if kf in acc[cam]:
                continue
            im = decode(m.ros_msg)
            if im is None:
                continue
            buf[cam].append((kf, im))
            n_msgs += 1
            if len(buf[cam]) >= a.batch:
                flush(cam)
        for cam in cams:
            flush(cam)
        for cam in cams:
            n = len(acc[cam])
            if n:
                idx = np.array(sorted(acc[cam]))
                D = np.stack([acc[cam][i] for i in idx]).astype(np.float32)
                full = np.full((len(need[cam]), D.shape[1]), np.nan, np.float32)
                full[idx] = D
                np.save(out / f"{cam}_dino_{src}.npy", full.astype(np.float16))
                day[cam].append(full)
            print(f"[{src}] {cam}: {n}/{len(need[cam])} keyframes found", flush=True)
        print(f"[{src}] {n_msgs} images used, {(time.time() - t0) / 60:.1f} min")

    # combined descriptor: normalised (w * night + sum of available day descriptors)
    for cam in cams:
        night = np.load(mp / f"{cam}_dino.npy").astype(np.float32)
        tot = a.night_weight * night
        for full in day[cam]:
            tot = tot + np.nan_to_num(full)
        tot /= np.linalg.norm(tot, axis=1, keepdims=True) + 1e-9
        np.save(out / f"{cam}_dino.npy", tot.astype(np.float16))

    # everything else: symlinks to the original map
    for f in mp.iterdir():
        dst = out / f.name
        if dst.exists() or dst.is_symlink() or f.name.endswith("_dino.npy"):
            continue
        os.symlink(os.path.relpath(f, out), dst)
    print(f"augmented map -> {out}  (only <cam>_dino.npy differs from {mp})")


if __name__ == "__main__":
    main()
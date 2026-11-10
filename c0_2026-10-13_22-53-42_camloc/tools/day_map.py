#!/usr/bin/env python3
"""
Map with BOTH night and synthetic-day keyframes, used together by the unchanged localizer.

Every map keyframe gets extra copies, one per synthetic day source (CycleGAN, Marigold, ...):
same pose, but SuperPoint keypoints/descriptors and the DINOv2 descriptor are computed on the
synthetic day image. The 3D point of a day keypoint is taken from the night map point of the
same keyframe that lies within --radius px (night->day translation keeps the geometry), so no
new triangulation is needed. At query time DINOv2 picks whichever copy LOOKS more similar
(night copy at night, day copy at day), and its points are matched as usual.

    python day_map.py --data data --map maps/colmap_7cam_kf2 \
        --mcap ../drive_data/cyclegan.mcap ../drive_data/marigold_v2.mcap \
        --out maps/colmap_7cam_kf2_nightday
"""
import argparse, json, re, shutil, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mcap_ros2.reader import read_ros2_messages

import featmap as F
from dino_desc import Dino
from dino_aug import IMG, stamp, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--map", default="maps/colmap_7cam_kf2")
    ap.add_argument("--mcap", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=3.0, help="px, day keypoint -> night 3D point")
    ap.add_argument("--max-kp", type=int, default=2048)
    ap.add_argument("--max-pts-per-kf", type=int, default=600)
    ap.add_argument("--tol-ms", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    data, mp, out = Path(a.data), Path(a.map), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((mp / "meta.json").read_text())
    cams = meta["cams"]
    index = pd.read_parquet(mp / "index.parquet")
    feats = F.Features(a.max_kp)
    dino = Dino()
    tol = int(a.tol_ms * 1e6)

    night = {}
    for cam in cams:
        z = np.load(mp / f"{cam}.npz")
        o = np.argsort(z["kf"], kind="stable")
        kf = z["kf"][o]
        n_kf = len(pd.read_parquet(mp / f"{cam}_kf.parquet"))
        night[cam] = dict(kp=z["kp"][o], desc=z["desc"][o], xyz=z["xyz"][o], score=z["score"][o], kf=kf,
                          start=np.searchsorted(kf, np.arange(n_kf)),
                          end=np.searchsorted(kf, np.arange(n_kf), side="right"), n_kf=n_kf)
    stamps = {c: index[index.cam == c].stamp_ns.values for c in cams}
    order = {c: np.argsort(stamps[c]) for c in cams}
    sorted_st = {c: stamps[c][order[c]] for c in cams}

    day = {c: [] for c in cams}            # per source: dict(kp, desc, xyz, score, kf, dino)
    for path in a.mcap:
        src, t0 = Path(path).stem, time.time()
        res = {c: dict(kp=[], desc=[], xyz=[], score=[], kf=[],
                       dino=np.full((night[c]["n_kf"], 0), np.nan)) for c in cams}
        dino_acc = {c: {} for c in cams}
        seen = {c: set() for c in cams}
        buf = {c: [] for c in cams}
        n_used, n_pts = 0, 0

        def flush(cam):
            nonlocal n_pts
            if not buf[cam]:
                return
            ks, ims = zip(*buf[cam])
            for k, d in zip(ks, dino(list(ims))):
                dino_acc[cam][k] = d
            nm = night[cam]
            for k, im in zip(ks, ims):
                q = feats.extract(im)
                s, e = nm["start"][k], nm["end"][k]
                if e <= s or not len(q["kp"]):
                    continue
                dist, j = cKDTree(nm["kp"][s:e]).query(q["kp"], distance_upper_bound=a.radius)
                ok = np.isfinite(dist)
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                idx = idx[np.argsort(-q["score"][idx])][:a.max_pts_per_kf]
                r = res[cam]
                r["kp"].append(q["kp"][idx]); r["desc"].append(q["desc"][idx])
                r["score"].append(q["score"][idx]); r["xyz"].append(nm["xyz"][s:e][j[idx]])
                r["kf"].append(np.full(len(idx), k)); n_pts += len(idx)
            buf[cam].clear()

        for m in read_ros2_messages(path):
            mi = IMG.search(m.channel.topic)
            if not mi or mi.group(1) not in night:
                continue
            cam = mi.group(1)
            s = stamp(m.ros_msg.header)
            arr = sorted_st[cam]
            jj = np.searchsorted(arr, s)
            cand = [x for x in (jj - 1, jj) if 0 <= x < len(arr) and abs(int(arr[x]) - s) <= tol]
            if not cand:
                continue
            k = int(order[cam][min(cand, key=lambda x: abs(int(arr[x]) - s))])
            if k in seen[cam]:
                continue
            im = decode(m.ros_msg)
            if im is None:
                continue
            seen[cam].add(k); buf[cam].append((k, im)); n_used += 1
            if len(buf[cam]) >= a.batch:
                flush(cam)
        for cam in cams:
            flush(cam)
            D = None
            if dino_acc[cam]:
                dim = len(next(iter(dino_acc[cam].values())))
                D = np.full((night[cam]["n_kf"], dim), np.nan, np.float32)
                for k, d in dino_acc[cam].items():
                    D[k] = d
            res[cam]["dino"] = D
            day[cam].append(res[cam])
            print(f"[{src}] {cam}: {len(seen[cam])}/{night[cam]['n_kf']} keyframes, "
                  f"{sum(len(x) for x in res[cam]['kf'])} day points with 3D", flush=True)
        print(f"[{src}] {n_used} images, {n_pts} day points, {(time.time() - t0) / 60:.1f} min")

    # ---- write the combined map: night keyframes first, then one copy per source
    new_index = []
    for cam in cams:
        nm, n_kf = night[cam], night[cam]["n_kf"]
        parts = {k: [nm[k]] for k in ("kp", "desc", "xyz", "score", "kf")}
        kt0 = pd.read_parquet(mp / f"{cam}_kf.parquet")
        kts, dinos = [kt0.assign(src="night")], [np.load(mp / f"{cam}_dino.npy").astype(np.float32)]
        sub = index[index.cam == cam]
        idx_parts = [sub]
        for s_i, r in enumerate(day[cam], start=1):
            off = s_i * n_kf
            if r["kf"]:
                for k in ("kp", "desc", "xyz", "score"):
                    parts[k].append(np.concatenate(r[k]))
                parts["kf"].append(np.concatenate(r["kf"]) + off)
            src = Path(a.mcap[s_i - 1]).stem
            kts.append(kt0.assign(src=src))
            D = r["dino"] if r["dino"] is not None else dinos[0].copy()
            D = np.where(np.isnan(D), dinos[0], D)          # missing day frame -> night descriptor
            dinos.append(D)
            idx_parts.append(sub)
        kf_all = np.concatenate(parts["kf"])
        np.savez(out / f"{cam}.npz", kp=np.concatenate(parts["kp"]).astype(np.float32),
                 desc=np.concatenate(parts["desc"]).astype(np.float16),
                 xyz=np.concatenate(parts["xyz"]).astype(np.float32),
                 score=np.concatenate(parts["score"]).astype(np.float32), kf=kf_all)
        kt = pd.concat(kts, ignore_index=True)
        kt["n_pts"] = np.bincount(kf_all, minlength=len(kt))
        kt.to_parquet(out / f"{cam}_kf.parquet")
        np.save(out / f"{cam}_dino.npy", np.concatenate(dinos).astype(np.float16))
        new_index.append(pd.concat(idx_parts, ignore_index=True))
        print(f"[{cam}] keyframes {len(kt)} ({len(day[cam])} day copies), points {len(kf_all):,}")
    pd.concat(new_index, ignore_index=True).to_parquet(out / "index.parquet")
    meta = dict(meta, source="night+synthetic-day", day_sources=[Path(p).stem for p in a.mcap])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
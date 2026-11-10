#!/usr/bin/env python3
"""
What is in a .ply point cloud, and is it in the same coordinate frame as our map / GT?

    pip install plyfile
    python ply_check.py --ply ../track_cloud_nested.ply --data data --map maps/colmap_7cam_kf2

Prints the fields, number of points and bounds; saves ply_check/top.png (cloud from above with
the track borders and the GT path) and ply_check/<cam>.jpg (cloud points projected into a
training image with the GT pose: if they lie on the walls / kerbs / poles, the frames agree).
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

import featmap as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--map", default="maps/colmap_7cam_kf2")
    ap.add_argument("--cams", nargs="+", default=["fr", "cl", "r"])
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--out", default="ply_check")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(exist_ok=True)

    ply = PlyData.read(a.ply)
    for el in ply.elements:
        print(f"element '{el.name}': {el.count} items, fields {[p.name for p in el.properties]}")
    v = ply["vertex"]
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    print(f"bounds x {xyz[:,0].min():.1f}..{xyz[:,0].max():.1f}  y {xyz[:,1].min():.1f}..{xyz[:,1].max():.1f}"
          f"  z {xyz[:,2].min():.1f}..{xyz[:,2].max():.1f}")
    if len(xyz) > a.max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), a.max_points, replace=False)]

    gt = pd.read_parquet(data / "gt_tf.parquet")
    print(f"GT path bounds  x {gt.x.min():.1f}..{gt.x.max():.1f}  y {gt.y.min():.1f}..{gt.y.max():.1f}"
          f"  z {gt.z.min():.1f}..{gt.z.max():.1f}")
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz[:, :2])
    d, _ = tree.query(gt[["x", "y"]].values[::50])
    print(f"GT path -> nearest cloud point (top view): median {np.median(d):.2f} m, max {d.max():.2f} m "
          f"(small = same frame and the cloud covers the track)")

    fig, ax = plt.subplots(figsize=(11, 11))
    s = xyz[np.random.default_rng(1).choice(len(xyz), min(300_000, len(xyz)), replace=False)]
    ax.scatter(s[:, 0], s[:, 1], s=0.05, c=s[:, 2], cmap="viridis")
    tb = data / "track_bounds.json"
    if tb.exists():
        for _, b in json.loads(tb.read_text()):
            b = np.asarray(b); ax.plot(b[:, 0], b[:, 1], "r", lw=0.6)
    ax.plot(gt.x[::20], gt.y[::20], "k", lw=0.5)
    ax.set_aspect("equal"); ax.set_title("cloud (colour = z), track borders red, GT path black")
    fig.savefig(out / "top.png", dpi=130, bbox_inches="tight")

    calib = json.loads((data / "calib.json").read_text())
    fr = pd.read_parquet(data / "frames.parquet")
    fr = fr[fr.gt_ok & (fr.lap == 4)]
    row_t = fr.stamp_ns.iloc[len(fr) // 3]
    for cam in a.cams:
        c = F.Cam(calib, cam)
        r = fr[(fr.cam == cam) & (fr.stamp_ns == row_t)]
        if r.empty:
            r = fr[fr.cam == cam].iloc[[len(fr[fr.cam == cam]) // 3]]
        r = r.iloc[0]
        T_cm = np.linalg.inv(F.T_map_base_row(r) @ c.T_bc)
        near = np.hypot(xyz[:, 0] - r.x, xyz[:, 1] - r.y) < 120
        Xc = xyz[near] @ T_cm[:3, :3].T + T_cm[:3, 3]
        Xc = Xc[(Xc[:, 2] > 1) & (Xc[:, 2] < 100)]
        img = cv2.imread(str(data / r.path))
        uv, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), c.K, c.D)
        uv = uv.reshape(-1, 2)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        col = cv2.applyColorMap(np.clip(Xc[ok, 2] / 60 * 255, 0, 255).astype(np.uint8)[:, None],
                                cv2.COLORMAP_TURBO)[:, 0]
        vis = (img * 0.6).astype(np.uint8)
        for (u, v_), cc in zip(uv[ok].astype(int), col):
            cv2.circle(vis, (u, v_), 1, tuple(int(x) for x in cc), -1)
        cv2.imwrite(str(out / f"{cam}.jpg"), vis)
        print(f"[{cam}] {ok.sum()} cloud points projected -> {out / (cam + '.jpg')}")
    print(f"top view -> {out / 'top.png'}")


if __name__ == "__main__":
    main()
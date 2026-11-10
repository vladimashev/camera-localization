#!/usr/bin/env python3
"""
Map with LiDAR geometry: same keyframes and SuperPoint descriptors as the COLMAP map, but the
3D position of each keypoint comes from the point cloud (depth looked up along its viewing ray)
instead of from triangulation. More points (also those COLMAP could not triangulate) and more
accurate ones. Output is in the same format, so featmap_mix.py / featmap.py use it unchanged.

    python lidar_map.py --data data --src maps/colmap_7cam_kf2 \
        --ply ~/camera-localization/track_cloud_nested.ply --out maps/lidar_7cam_kf2

How: every keyframe pose (GT, training laps only, like the COLMAP map) -> all cloud points
projected on the GPU into a depth buffer (min depth per pixel, then a small min filter so the
nearest surface wins through gaps of the sparse cloud) -> depth at each keypoint -> 3D point.
Keypoints on the own car body (data/masks/<cam>.png) are dropped: the cloud has no car.
Prints how well the LiDAR points agree with the COLMAP points (a check of the alignment).
"""
import argparse, json, shutil, time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from plyfile import PlyData

import featmap as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_cloud(path, zmin, zmax):
    v = PlyData.read(path)["vertex"]
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float32)
    xyz = xyz[(xyz[:, 2] > zmin) & (xyz[:, 2] < zmax)]
    print(f"cloud: {len(xyz):,} points")
    return torch.from_numpy(xyz).to(DEV)


class DepthCam:
    def __init__(self, cam, W, H, max_depth, win):
        self.c, self.W, self.H, self.max_depth, self.win = cam, W, H, max_depth, win
        K, D = cam.K, np.r_[cam.D, np.zeros(5)][:5]
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = [float(x) for x in D]
        # valid range of undistorted normalized coordinates (avoids wrap-around of the model)
        u, v = np.meshgrid(np.linspace(0, W - 1, 40), np.linspace(0, H - 1, 20))
        n = cam.norm(np.c_[u.ravel(), v.ravel()])
        self.nb = (n[:, 0].min() - 0.05, n[:, 0].max() + 0.05, n[:, 1].min() - 0.05, n[:, 1].max() + 0.05)

    @torch.no_grad()
    def depth(self, X, T_cm):
        R = torch.as_tensor(T_cm[:3, :3], dtype=torch.float32, device=DEV)
        t = torch.as_tensor(T_cm[:3, 3], dtype=torch.float32, device=DEV)
        Xc = X @ R.T + t
        z = Xc[:, 2]
        m = (z > 0.5) & (z < self.max_depth)
        Xc, z = Xc[m], z[m]
        x, y = Xc[:, 0] / z, Xc[:, 1] / z
        x0, x1, y0, y1 = self.nb
        m = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        x, y, z = x[m], y[m], z[m]
        r2 = x * x + y * y
        rad = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        xd = x * rad + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * rad + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        u = torch.round(self.fx * xd + self.cx).long()
        v = torch.round(self.fy * yd + self.cy).long()
        m = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        buf = torch.full((self.H * self.W,), float("inf"), device=DEV)
        buf.scatter_reduce_(0, v[m] * self.W + u[m], z[m], reduce="amin")
        buf = buf.view(1, 1, self.H, self.W)
        # nearest surface in a small window: background points must not shine through gaps
        buf = -torch.nn.functional.max_pool2d(-buf, self.win, stride=1, padding=self.win // 2)
        return buf[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--src", default="maps/colmap_7cam_kf2", help="COLMAP map (keyframes, features.h5)")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default="maps/lidar_7cam_kf2")
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--win", type=int, default=5, help="px, min-depth window")
    ap.add_argument("--max-pts-per-kf", type=int, default=1000)
    ap.add_argument("--keep-colmap", action="store_true",
                    help="keep COLMAP points where the cloud has no depth (sky, far)")
    ap.add_argument("--zmin", type=float, default=-20); ap.add_argument("--zmax", type=float, default=60)
    a = ap.parse_args()
    data, src, out = Path(a.data), Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    cams = meta["cams"]
    calib = json.loads((data / "calib.json").read_text())
    index = pd.read_parquet(src / "index.parquet")
    frames = pd.read_parquet(data / "frames.parquet").set_index(["cam", "stamp_ns"])
    feats_path = src / "features.h5"
    if not feats_path.exists():
        raise SystemExit(f"{feats_path} missing: it holds all keypoints/descriptors of the keyframes")
    X = load_cloud(a.ply, a.zmin, a.zmax)

    t0 = time.time()
    with h5py.File(feats_path, "r") as h:
        for cam in cams:
            c = F.Cam(calib, cam)
            cc = calib["cameras"][cam]
            dc = DepthCam(c, cc["width"], cc["height"], a.max_depth, a.win)
            mask = F.load_mask(data, cam)
            old = np.load(src / f"{cam}.npz")
            okf, oxyz, okp = old["kf"], old["xyz"], old["kp"]
            sub = index[index.cam == cam].reset_index(drop=True)
            st = {"kp": [], "desc": [], "xyz": [], "score": [], "kf": []}
            agree, n_lidar, n_colmap_only = [], 0, 0
            for k, r in enumerate(sub.itertuples()):
                if r.name not in h:
                    continue
                g = h[r.name]
                kp = g["keypoints"][()].astype(np.float32)
                desc = g["descriptors"][()].T.astype(np.float16)
                sc = g["scores"][()].astype(np.float32)
                fr = frames.loc[(cam, r.stamp_ns)]
                T_mc = F.T_map_base_row(fr) @ c.T_bc
                D = dc.depth(X, np.linalg.inv(T_mc))
                u = np.clip(np.round(kp[:, 0]).astype(int), 0, dc.W - 1)
                v = np.clip(np.round(kp[:, 1]).astype(int), 0, dc.H - 1)
                d = D[torch.as_tensor(v, device=DEV), torch.as_tensor(u, device=DEV)].cpu().numpy()
                ok = np.isfinite(d)
                if mask is not None:
                    ok &= mask[v, u] > 0
                n = c.norm(kp)
                Xc = np.c_[n, np.ones(len(n))] * d[:, None]
                Xm = np.full((len(kp), 3), np.nan, np.float32)
                Xm[ok] = (Xc[ok] @ T_mc[:3, :3].T + T_mc[:3, 3]).astype(np.float32)
                # compare with COLMAP points of this keyframe (same pixel)
                sel = okf == k
                if sel.any():
                    tree = {(int(round(p[0])), int(round(p[1]))): i for i, p in enumerate(kp)}
                    for p, xo in zip(okp[sel], oxyz[sel]):
                        i = tree.get((int(round(p[0])), int(round(p[1]))))
                        if i is None:
                            continue
                        if ok[i]:
                            agree.append(np.linalg.norm(Xm[i] - xo))
                        elif a.keep_colmap:
                            Xm[i] = xo; ok[i] = True; n_colmap_only += 1
                n_lidar += int(np.isfinite(Xm[:, 0]).sum())
                idx = np.flatnonzero(ok & np.isfinite(Xm[:, 0]))
                if not len(idx):
                    continue
                idx = idx[np.argsort(-sc[idx])][:a.max_pts_per_kf]
                st["kp"].append(kp[idx]); st["desc"].append(desc[idx]); st["score"].append(sc[idx])
                st["xyz"].append(Xm[idx]); st["kf"].append(np.full(len(idx), k))
                if k % 1000 == 0:
                    print(f"  [{cam}] kf {k}/{len(sub)}  {time.time() - t0:.0f}s", flush=True)
            kf = np.concatenate(st["kf"])
            np.savez(out / f"{cam}.npz", kp=np.concatenate(st["kp"]), desc=np.concatenate(st["desc"]),
                     xyz=np.concatenate(st["xyz"]), score=np.concatenate(st["score"]), kf=kf)
            kt = pd.read_parquet(src / f"{cam}_kf.parquet")
            kt["n_pts"] = np.bincount(kf, minlength=len(kt))
            kt.to_parquet(out / f"{cam}_kf.parquet")
            ag = np.array(agree)
            print(f"[{cam}] {len(kf):,} map points (COLMAP map had {len(okf):,}), "
                  f"median {int(kt.n_pts.median())} per keyframe"
                  + (f", {n_colmap_only:,} kept from COLMAP" if a.keep_colmap else ""))
            if len(ag):
                print(f"      LiDAR vs COLMAP 3D point: median {np.median(ag):.3f} m, "
                      f"p90 {np.percentile(ag, 90):.3f} m ({len(ag):,} shared keypoints)")
    for f in ["index.parquet"] + [p.name for p in src.glob("*_dino*.npy")]:
        shutil.copy(src / f, out / f)
    meta = dict(meta, source="lidar", src_map=str(src), ply=str(a.ply))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
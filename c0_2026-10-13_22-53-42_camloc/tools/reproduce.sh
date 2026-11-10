#!/usr/bin/env bash
# Full pipeline from the raw recordings to all results. Run from tools/.
# Every step is skipped if its output already exists, so the script can be re-run safely.
#
#   conda activate camloc
#   bash reproduce.sh                 # everything
#   bash reproduce.sh night           # only up to the night validation
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_MCAP=../c0_2026-10-13_22-53-42_camloc_0.mcap
BAG_MCAP=../rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap
MAP=maps/colmap_7cam_kf2
VAL_LAPS="3 6"                          # held out of map and calibration
MAP_LAPS="0 1 2 4 5 7 8"
CAMS="fl fr cl cr r rl rr"
REQ=../../requirements.txt             # requirements.txt in the project root (~/camera-localization)
STAGE=${1:-all}

step() { echo; echo "=== $*"; }

# ---------------------------------------------------------------- 0. checks
if [ -f "$REQ" ]; then pip install -q -r "$REQ"; else echo "WARNING: $REQ not found, skipping pip install"; fi
[ -f gt.csv ] || { echo "gt.csv missing: run extract_gt.py from the starter kit (tools/)"; exit 1; }

# ---------------------------------------------------------------- 1. training data
step "1. index the training mcap (images, calibration, GT, laps)"
[ -f data/frames.parquet ] || python lap_index.py extract --mcap $TRAIN_MCAP --laps laps.csv --out data/

step "2. car state (vx_mps, vy_mps, wz_radps)"
[ -f data/state.parquet ] || python extract_state.py --mcap $TRAIN_MCAP --data data/

step "3. track borders (data/track_bounds.json, for plots)"
[ -f data/track_bounds.json ] || python plot_laps.py --mcap $TRAIN_MCAP --data data/

step "4. motion-model calibration without the validation laps"
[ -f data/motion_calib.json ] || python calib_motion.py --data data/ --exclude-laps $VAL_LAPS \
    --out data/motion_calib.json

# ---------------------------------------------------------------- 2. map
step "5. COLMAP map: 7 cameras, keyframes every 2 m, without the validation laps (~1 h)"
[ -f $MAP/meta.json ] || python -u colmap_map.py build --data data/ --laps $MAP_LAPS \
    --cams $CAMS --kf-dist 2 --num-pairs 15 --out $MAP

step "6. DINOv2 descriptors of the map keyframes"
[ "$(ls $MAP/*_dino.npy 2>/dev/null | wc -l)" -ge 7 ] || python dino_desc.py --data data/ --map $MAP

step "7. brightness cache of the map: must be computed from the TRAINING data folder"
python - << 'PY'
import glob, numpy as np, os
bad = []
for f in glob.glob("maps/colmap_7cam_kf2/light_*.npz"):
    z = np.load(f)
    if any(np.issubdtype(z[k].dtype, np.floating) and np.isnan(z[k]).any() for k in z.files):
        bad.append(f)
for f in bad:
    os.remove(f); print("removed broken cache", f)
print("cache ok" if not bad else "cache will be rebuilt by the next night run")
PY

# ---------------------------------------------------------------- 3. night validation
step "8. night validation, laps $VAL_LAPS (featmap_mix)"
[ -f mix_v7.csv ] || python featmap_mix.py validate --data data/ --map $MAP --laps $VAL_LAPS \
    --state data/state.parquet --dino --out mix_v7.csv

step "9. official metrics"
python - << 'PY'
import pandas as pd
gt = pd.read_csv("gt.csv"); laps = pd.read_csv("data/laps.csv"); t0 = gt.timestamp_ns.min()
for lap in (3, 6):
    r = laps[laps.lap == lap].iloc[0]
    lo, hi = t0 + int(r.t_start * 1e9), t0 + int(r.t_end * 1e9)
    gt[(gt.timestamp_ns >= lo) & (gt.timestamp_ns <= hi)].to_csv(f"gt_lap{lap}.csv", index=False)
PY
for lap in $VAL_LAPS; do
    echo "--- lap $lap"; python3 evaluate.py gt_lap$lap.csv mix_v7.csv
done
[ "$STAGE" == "night" ] && exit 0

# ---------------------------------------------------------------- 4. day bag (no GT)
step "10. day bag -> data_bag/ (bag lap 99 + training index for the map keyframes)"
[ -f data_bag/frames.parquet ] || python convert_bag.py --mcap $BAG_MCAP --out data_bag \
    --train-data data --with-train-index --lap 99

step "11. first pose of the bag"
# DIAGNOSTIC: from the shape of the whole lap (non-causal). If a first pose is given,
# use instead:  python init_pose.py --data data_bag --map $MAP --lap 99 --x X --y Y --yaw YAW --check-sets 0
[ -f data_bag/first_pose.json ] || python init_pose.py --data data_bag --map $MAP --lap 99 \
    --shape --train-data data --check-sets 0

step "12. day bag localization (featmap_mix); errors are meaningless here, there is no GT"
[ -f bag_run.csv ] || python featmap_mix.py validate --data data_bag --map $MAP --laps 99 \
    --state data_bag/state.parquet --dino --motion-calib data_bag/motion_calib.json --out bag_run.csv

step "13. plots"
python plot_bag.py --data data_bag --run bag_run.csv --odom-scale 1.0043
python plot_traj.py --runs mix_v7.csv:night_K3_K6 --gt data/gt_tf.parquet --gt-laps $VAL_LAPS \
    --laps-csv data/laps.csv --bounds data/track_bounds.json --out traj_night.png
echo; echo "done"
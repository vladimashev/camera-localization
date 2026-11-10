#!/usr/bin/env bash
# Leave-laps-out cross-validation, fully unattended.
#
# For every fold (a set of held-out laps):
#   1. motion calibration on the other laps            -> data/motion_calib_<fold>.json
#   2. COLMAP map from the other laps (7 cams, 2 m)    -> maps/cv_<fold>/
#   3. DINOv2 descriptors of the map keyframes
#   4. localization on the held-out laps                -> cv/<fold>.csv  (+ cv/<fold>.log)
# then cv_summary.py scores every held-out lap with the official evaluate.py.
#
# Resumable: finished steps are skipped, so after a crash just start it again.
# Folds run PARALLEL at a time.
# Run detached:   nohup bash run_cv.sh > cv_master.log 2>&1 &
# Watch:          tail -f cv/cv.log
set -uo pipefail
cd "$(dirname "$0")"

# ------------------------------------------------------------------ settings
DATA=data
ALL_LAPS="0 1 2 3 4 5 6 7 8"           # 0 and 8 (pit in/out) always stay in training
# held-out laps per fold: all 21 pairs of the full laps 1..7 ("3 6" reuses the existing map)
FOLDS=("1 2" "1 3" "1 4" "1 5" "1 6" "1 7" "2 3" "2 4" "2 5" "2 6" "2 7"
       "3 4" "3 5" "3 6" "3 7" "4 5" "4 6" "4 7" "5 6" "5 7" "6 7")
PARALLEL=3                             # folds running at the same time
CAMS="fl fr cl cr r rl rr"
LOC=featmap_mix.py                     # localizer to validate
LOC_ARGS="--dino"                      # extra validate args (same as your best run)
REUSE_36=maps/colmap_7cam_kf2          # existing map built without laps 3 and 6
CLEANUP=1                              # 1: delete features.h5 / matches.h5 / database.db after a fold
OUT=cv
# ----------------------------------------------------------------------------

mkdir -p "$OUT" maps
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/cv.log"; }

[ -f gt.csv ] || { log "gt.csv missing (extract_gt.py from the starter kit)"; exit 1; }
[ -f "$DATA/state.parquet" ] || { log "$DATA/state.parquet missing (extract_state.py)"; exit 1; }

# does the localizer take --motion-calib? (otherwise it may silently use a calibration
# fitted WITH the held-out laps -> leakage)
if python "$LOC" validate -h 2>/dev/null | grep -q -- "--motion-calib"; then
    HAS_CALIB=1
else
    HAS_CALIB=0
    log "WARNING: $LOC has no --motion-calib option. Make sure it does not load a calibration"
    log "         fitted on the held-out laps, otherwise the results are optimistic."
fi

run_fold() {
    local VAL="$1"
    local TAG="val_${VAL// /_}"
    TRAIN=$(for l in $ALL_LAPS; do [[ " $VAL " == *" $l "* ]] || printf "%s " "$l"; done)
    MAP="maps/cv_$TAG"
    CAL="$DATA/motion_calib_$TAG.json"
    FLOG="$OUT/$TAG.log"
    T0=$(date +%s)
    log "===== fold $TAG: train on laps [ $TRAIN], validate on [ $VAL ]"

    # 1. motion calibration without the held-out laps
    if [ ! -f "$CAL" ]; then
        log "[$TAG] motion calibration"
        python calib_motion.py --data "$DATA" --exclude-laps $VAL --out "$CAL" >> "$FLOG" 2>&1 \
            || { log "[$TAG] calibration FAILED, see $FLOG"; return 1; }
    fi

    # 2. map without the held-out laps (reuse the existing one for 3+6)
    if [ "$VAL" == "3 6" ] && [ -d "$REUSE_36" ] && [ ! -e "$MAP" ]; then
        ln -s "$(basename "$REUSE_36")" "$MAP"
        log "[$TAG] reusing $REUSE_36"
    fi
    if [ ! -f "$MAP/meta.json" ]; then
        log "[$TAG] COLMAP map (this is the long step, ~1 h)"
        python -u colmap_map.py build --data "$DATA" --laps $TRAIN --cams $CAMS \
            --kf-dist 2 --num-pairs 15 --out "$MAP" >> "$FLOG" 2>&1 \
            || { log "[$TAG] map build FAILED, see $FLOG"; return 1; }
    fi

    # 3. DINOv2 descriptors
    n_dino=$(ls "$MAP"/*_dino.npy 2>/dev/null | wc -l)
    if [ "$n_dino" -lt "$(echo $CAMS | wc -w)" ]; then
        log "[$TAG] DINOv2 descriptors"
        python -u dino_desc.py --data "$DATA" --map "$MAP" >> "$FLOG" 2>&1 \
            || { log "[$TAG] DINOv2 FAILED, see $FLOG"; return 1; }
    fi

    # 4. localization on the held-out laps
    CSV="$OUT/$TAG.csv"
    if [ ! -f "$CSV" ]; then
        log "[$TAG] localization on laps [ $VAL ]"
        CALARG=""; [ "$HAS_CALIB" == 1 ] && CALARG="--motion-calib $CAL"
        python -u "$LOC" validate --data "$DATA" --map "$MAP" --laps $VAL \
            --state "$DATA/state.parquet" $CALARG $LOC_ARGS --out "$CSV" >> "$FLOG" 2>&1 \
            || { log "[$TAG] localization FAILED, see $FLOG"; return 1; }
        grep -E "per frame: mean|REAL-TIME|^ +all:|^ +lap [0-9]+: pos" "$FLOG" | tail -n 6 \
            | while read -r line; do log "[$TAG] $line"; done
    fi

    if [ "$CLEANUP" == 1 ] && [ ! -L "$MAP" ]; then
        rm -f "$MAP/features.h5" "$MAP/matches.h5" "$MAP/sfm/database.db"
        log "[$TAG] removed intermediate COLMAP files"
    fi
    log "[$TAG] done in $(( ($(date +%s) - T0) / 60 )) min"
}

for VAL in "${FOLDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
    run_fold "$VAL" &
    sleep 60                           # stagger the starts (feature extraction is I/O heavy)
done
wait
log "NOTE: folds ran $PARALLEL at a time, so the runtime numbers in the fold logs are inflated;"
log "      measure the time per frame on a separate run with nothing else on the GPU."

log "===== scoring all folds with evaluate.py"
python cv_summary.py --gt gt.csv --laps-csv "$DATA/laps.csv" --dir "$OUT" 2>&1 | tee -a "$OUT/cv.log"
log "===== finished"
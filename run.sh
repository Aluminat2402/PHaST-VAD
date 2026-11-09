#!/usr/bin/env bash
# PHaST-VAD launcher. Dataset locations come from environment variables (nothing is hard-coded):
#   UCF_VIDEOS, UCF_ANN     UCF-Crime test videos dir + Temporal_Anomaly_Annotation_for_Testing_Videos.txt
#   XD_VIDEOS,  XD_ANN      XD-Violence test videos dir + annotation in the same format (see README)
#   UBN_ROOT                UBnormal root (Scene1 ... Scene29); the annotation is written by prepare_ubnormal.py
#   MSAD_ROOT               MSAD root; the annotation is written by prepare_msad.py
#   HF_HUB_CACHE            optional HuggingFace cache;  OUT  output root (default: outputs);  WITH_Q=1 adds the q probe
# Usage:
#   bash run.sh offline ucf|xd|ubn|msad GPU [GPU ...]   offline scoring, one shard per GPU, in background
#   bash run.sh stream  ucf|xd|ubn|msad GPU [GPU ...]   wall-clock streaming replay (one stream per GPU), in background
#   bash run.sh eval    ucf|xd|ubn|msad|msad_ii [SCORE] [offline|stream]   metrics (SCORE: PH QH QPH p a h pa q)
#   bash run.sh latency ucf|xd|ubn|msad                 latency summary of the streaming run
#   bash run.sh smoke   GPU VIDEO.mp4                   4 windows offline + 6 windows of streaming replay on one video
set -euo pipefail
[ "${1:-}" = smoke ] && [ $# -eq 3 ] && set -- "$1" "$2" "$(realpath "$3")"
cd "$(dirname "$0")"
OUT=${OUT:-outputs}
Q=(); [ "${WITH_Q:-0}" = 1 ] && Q=(--with-q)
mkdir -p "$OUT/logs"

need() { [ -n "${!1:-}" ] || { echo "set $1 (see the header of $0)" >&2; exit 1; }; }
data_args() {
  case "$1" in
    ucf)  need UCF_VIDEOS; need UCF_ANN; echo "--video-dir $UCF_VIDEOS --annotation $UCF_ANN" ;;
    xd)   need XD_VIDEOS; need XD_ANN; echo "--video-dir $XD_VIDEOS --annotation $XD_ANN" ;;
    ubn)  need UBN_ROOT; [ -f "$OUT/ubn/annotation.txt" ] || python prepare_ubnormal.py "$UBN_ROOT" --output "$OUT/ubn/annotation.txt" >&2
          echo "--video-dir $UBN_ROOT --annotation $OUT/ubn/annotation.txt" ;;
    msad) need MSAD_ROOT; [ -f "$OUT/msad/msad_test_annotation.txt" ] || python prepare_msad.py "$MSAD_ROOT" --out "$OUT/msad" >&2
          echo "--video-dir $MSAD_ROOT --annotation $OUT/msad/msad_test_annotation.txt" ;;
    *)    echo "unknown dataset $1 (ucf|xd|ubn|msad)" >&2; exit 1 ;;
  esac
}

case "${1:-}" in
  offline|stream)
    [ $# -ge 3 ] || { echo "usage: bash $0 $1 DATASET GPU [GPU ...]" >&2; exit 1; }
    mode=$1; ds=$2; shift 2
    args=$(data_args "$ds"); script=score_videos.py; [ "$mode" = stream ] && script="stream.py --quiet"
    shards=$#; i=0
    for gpu in "$@"; do
      log="$OUT/logs/${ds}_${mode}_shard$i.log"
      CUDA_VISIBLE_DEVICES=$gpu nohup python -u $script $args --output-dir "$OUT/$ds/$mode" ${Q[@]+"${Q[@]}"} \
        --num-shards "$shards" --shard-index "$i" > "$log" 2>&1 &
      echo "GPU $gpu  PID $!  shard $i/$shards -> $log"
      i=$((i + 1))
    done ;;
  eval)
    [ $# -ge 2 ] || { echo "usage: bash $0 eval DATASET [SCORE] [offline|stream]" >&2; exit 1; }
    ds=$2; score=${3:-PH}; mode=${4:-offline}; extra=()
    if [ "$ds" = msad_ii ]; then ds=msad; extra=(--video-list "$OUT/msad/msad_protocol_ii.txt"); fi
    python evaluate.py "$OUT/$ds/$mode/scores" --score "$score" ${extra[@]+"${extra[@]}"} --output "$OUT/$ds/$mode/metrics_${2}_${score}.json" ;;
  latency)
    [ $# -eq 2 ] || { echo "usage: bash $0 latency DATASET" >&2; exit 1; }
    python stream.py --summary --output-dir "$OUT/$2/stream" ;;
  smoke)
    [ $# -eq 3 ] || { echo "usage: bash $0 smoke GPU VIDEO.mp4" >&2; exit 1; }
    dir=$(dirname "$3"); name=$(basename "$3")
    CUDA_VISIBLE_DEVICES=$2 python -u score_videos.py --video-dir "$dir" --only-video "$name" --output-dir "$OUT/smoke/offline" \
      --max-windows 4 --check 1 ${Q[@]+"${Q[@]}"}
    CUDA_VISIBLE_DEVICES=$2 python -u stream.py --video-dir "$dir" --only-video "$name" --output-dir "$OUT/smoke/stream" \
      --max-windows 6 ${Q[@]+"${Q[@]}"} ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac

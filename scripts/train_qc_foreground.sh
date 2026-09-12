#!/usr/bin/env bash
set -e
BASE=/mnt/d/AI/piper-suite
NAME=piper-training-qc
TARGET=3804
QUALITY=medium
BATCH=12
DATASET="$BASE/datasets/marie_line_fr_qc"
SIWIS_CKPT="$BASE/pretrained/fr_fr_siwis_medium/epoch=3304-step=2050940.ckpt"
LOG="/mnt/d/AI/piper-suite/training_qc.log"

mkdir -p "$BASE/work_dir_qc" "$BASE/training_logs_qc"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "=== launching training (foreground, logged to $LOG) ==="
docker run --name "$NAME" --shm-size=8g --gpus all -w /app \
  -e MAX_EPOCHS="$TARGET" \
  -e QUALITY="$QUALITY" \
  -e BATCH_SIZE="$BATCH" \
  -e PYTHONPATH=/app \
  -e PIPER_LANGUAGE_CODE=fr \
  -v "$DATASET:/app/input_dataset" \
  -v "$BASE/work_dir_qc:/app/piper_training_dir" \
  -v "$BASE/training_logs_qc:/app/lightning_logs" \
  -v "$BASE/preprocess.sh:/app/preprocess.sh" \
  -v "$BASE/train.sh:/app/train.sh" \
  -v "$BASE/sitecustomize.py:/app/sitecustomize.py" \
  -v "$SIWIS_CKPT:/app/checkpoint.ckpt" \
  -v piper-datasetwork-qc:/app/dataset_work \
  chatterbox-piper:nightly-cu128-sm120 \
  /bin/bash -c "chmod +x /app/*.sh && /app/preprocess.sh && /app/train.sh" \
  > "$LOG" 2>&1 &
DPID=$!

while kill -0 "$DPID" 2>/dev/null; do
  sleep 45
  echo "--- progress @ $(date '+%T') ---"
  ls "$BASE/work_dir_qc"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | tail -1
done

wait "$DPID"; RC=$?
echo "=== CONTAINER FINISHED (rc=$RC) ==="
tail -25 "$LOG"
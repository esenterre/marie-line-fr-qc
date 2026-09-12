#!/usr/bin/env bash
set -e

BASE=/mnt/d/AI/piper-suite
VOICE="${VOICE:-marie_line_fr_qc}"
WORKDIR="$BASE/work_dir_qc"
LOGSDIR="$BASE/training_logs_qc"

echo "=== Export ONNX (QC) ==="

CKPT=$(find "$WORKDIR" "$LOGDIR" -name "*.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint found in $WORKDIR or $LOGDIR" >&2
  exit 1
fi
echo "Selected checkpoint: $CKPT"

CKPT_DIR=$(dirname "$CKPT")
CKPT_NAME=$(basename "$CKPT")
OUT_NAME="${CKPT_NAME%.ckpt}.onnx"

CONFIG=""
for c in "$WORKDIR/config.json" "$CKPT_DIR/config.json" "$LOGDIR/config.json"; do
  if [ -f "$c" ]; then CONFIG="$c"; break; fi
done
echo "Config: $CONFIG"

mkdir -p "$BASE/exports"

docker run --gpus all --rm -w /app \
  -v "$CKPT_DIR:/app/export_source" \
  -v "$BASE/sitecustomize.py:/app/sitecustomize.py" \
  -e PYTHONPATH=/app \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e PIPER_TORCH_ONNX_DYNAMO=0 \
  chatterbox-piper:nightly-cu128-sm120 \
  /bin/bash -lc "python3 -c 'import onnxscript' 2>/dev/null || python3 -m pip install -q onnx onnxscript; python3 -m piper_train.export_onnx /app/export_source/$CKPT_NAME /app/export_source/$OUT_NAME"

if [ ! -f "$CKPT_DIR/$OUT_NAME" ]; then
  echo "ERROR: ONNX export produced no file" >&2
  exit 1
fi

DST="$BASE/exports/${VOICE}-medium.onnx"
DST_JSON="$BASE/exports/${VOICE}-medium.onnx.json"
cp "$CKPT_DIR/$OUT_NAME" "$DST"
if [ -n "$CONFIG" ] && [ -f "$CONFIG" ]; then
  cp "$CONFIG" "$DST_JSON"
fi

echo "Exported:"
ls -la "$DST" "$DST_JSON" 2>/dev/null || echo "(json missing)"

echo "=== Export finished ==="
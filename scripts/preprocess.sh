#!/usr/bin/env bash
set -e
set -x

# Define dataset paths
# Expecting input mounted at /app/input_dataset (which maps to datasets/marie_line_fr)
INPUT_DIR="/app/input_dataset"
WORK_DIR="/app/dataset_work"
OUTPUT_DIR="/app/piper_training_dir"
READY_MARKER="$WORK_DIR/.piper_dataset_ready"

echo "Starting Preprocessing..."
echo "Input: $INPUT_DIR"

# Copy the frozen dataset into persistent work dir ONCE (survives relays via
# the named volume piper-datasetwork). rsync over the 9p mount is fragile, so
# avoid repeating it; if it still fails, retry a few times.
if [ ! -f "$READY_MARKER" ]; then
  echo "First-run dataset copy..."
  mkdir -p "$WORK_DIR"
  for attempt in 1 2 3 4 5; do
    echo "rsync attempt $attempt"
    if rsync -r --delete "$INPUT_DIR/" "$WORK_DIR/"; then
      touch "$READY_MARKER"
      break
    fi
    echo "rsync attempt $attempt failed, retrying in 5s..."
    sleep 5
  done
  [ -f "$READY_MARKER" ] || { echo "ERROR: dataset copy failed after 5 attempts"; exit 1; }
else
  echo "Dataset work cache present, skipping copy"
fi

# 2. Reduce volume to avoid clipping (Sox)
# Process all wav files in the work directory
echo "Normalizing audio..."
find "$WORK_DIR" -type f -name "*.wav" -exec sox -v 0.95 {} -r 22050 -t wav {}.22050tmp \;
find "$WORK_DIR" -type f -name "*.wav" -exec mv {}.22050tmp {} \;

# 3. Piper Preprocess
echo "Running Piper Preprocess..."
# Note: --dataset-format ljspeech is standard for CSVs with id|text format
python3 -m piper_train.preprocess \
  --language fr \
  --input-dir "$WORK_DIR/" \
  --output-dir "$OUTPUT_DIR/" \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050 \
  --max-workers 1

echo "Preprocessing Complete. Data ready in $OUTPUT_DIR"
#!/usr/bin/env bash
set -e
set -x

DATASET_DIR="/app/piper_training_dir"
# Output for lightning logs
LOGS_DIR="/app/training_logs"

# Checkpoint logic (Optional)
# If you have a checkpoint, mount it to /app/checkpoint.ckpt
CKPT_ARGS=""
NEWEST_CKPT=$(ls -t /app/piper_training_dir/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
if [ -n "$NEWEST_CKPT" ]; then
    CKPT_ARGS="--resume_from_checkpoint $NEWEST_CKPT"
    export PIPER_TORCH_LOAD_WEIGHTS_ONLY=0
    echo "Resuming from newest checkpoint: $NEWEST_CKPT"
elif [ -f "/app/checkpoint.ckpt" ]; then
    # Guard against Git LFS pointer files or other placeholder files.
    # A real Lightning checkpoint is typically hundreds of MB.
    CKPT_SIZE_BYTES=$(stat -c%s "/app/checkpoint.ckpt" 2>/dev/null || echo 0)
    if [ "$CKPT_SIZE_BYTES" -lt 1048576 ]; then
        echo "WARNING: /app/checkpoint.ckpt is only ${CKPT_SIZE_BYTES} bytes; skipping resume."
        echo "If this came from Git LFS, run: git lfs pull"
        CKPT_ARGS=""
    else
        CKPT_ARGS="--resume_from_checkpoint /app/checkpoint.ckpt"
        # Allow loading older Lightning checkpoints with newer torch (PyTorch>=2.6 weights_only change).
        # SECURITY: only safe if you trust the checkpoint file.
        export PIPER_TORCH_LOAD_WEIGHTS_ONLY=0
        echo "Resuming from checkpoint..."
    fi
fi

# Defaults (can be overridden by -e in Docker)
MAX_EPOCHS=${MAX_EPOCHS:-3000}
BATCH_SIZE=${BATCH_SIZE:-12}
QUALITY=${QUALITY:-high}

echo "Starting Training (Epochs: $MAX_EPOCHS, Batch: $BATCH_SIZE, Quality: $QUALITY)..."

python3 -m piper_train \
    --dataset-dir "$DATASET_DIR" \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size "$BATCH_SIZE" \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs "$MAX_EPOCHS" \
    --checkpoint-epochs 1 \
    --quality "$QUALITY" \
    --precision 32 \
    $CKPT_ARGS

echo "Training Finished."

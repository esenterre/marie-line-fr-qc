#!/usr/bin/env bash
set -e
SRC=/app/f5gen/batch300
DST=/mnt/d/AI/piper-suite/datasets/marie_line_fr_qc
mkdir -p "$DST/wavs"

echo "--- durations ---"
docker run --rm -v /app/f5gen:/app/f5gen f5tts bash -lc '
python - <<EOF
import glob, soundfile as sf
fs = sorted(glob.glob("/app/f5gen/batch300/*.wav"))
d = [sf.info(f).duration for f in fs]
print("n=%d min=%.2f max=%.2f mean=%.2f" % (len(d), min(d), max(d), sum(d)/len(d)))
EOF'

echo "--- copy wavs ---"
cp -f "$SRC"/qc_*.wav "$DST/wavs/"
ls "$DST/wavs" | wc -l

echo "--- metadata.csv ---"
: > "$DST/metadata.csv"
while IFS='|' read -r name text; do
  [ -z "$text" ] && continue
  echo "$(basename "$name")|$text" >> "$DST/metadata.csv"
done < "$SRC/manifest.txt"
wc -l "$DST/metadata.csv"
head -3 "$DST/metadata.csv"
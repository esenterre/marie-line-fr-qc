[🇫🇷 Français](README.md) · [🇬🇧 English](README.en.md)

# French voice with a Quebec accent for Home Assistant (Piper)

A **French text-to-speech voice with a Quebec accent** for [Home Assistant](https://www.home-assistant.io/) through the [Piper add-on](https://github.com/home-assistant/addons/tree/master/piper), built for **personal home-automation use**.

The starting point was a **reference voice file** (a synthetic voice): the clip was used to fine-tune **F5-TTS Quebec** (`model_quebec_v1`) to generate ~300 Quebec-style sentences, which then served as the dataset to **retrain Piper** (a VITS model) from the open-source `fr_fr_siwis_medium` checkpoint. The final result is far enough from the source to be considered a distinct voice.

This repository contains the **full recipe**, the **scripts**, the **phrase list**, **audio samples** and the **final voice** (`.onnx` + `.onnx.json`). The reference audio, the 300 generated `.wav` files and the training checkpoints are **not** included.

## Listen before installing

Three samples generated with the final voice (in the `samples/` folder):

| File | Text |
|---|---|
| `samples/phrase1_qc.wav` | “Y fait vraiment beau aujourd'hui, on va jaser au bord du fleuve pis prendre un café.” |
| `samples/phrase2_qc.wav` | “Allume la lumière de la cuisine pis ferme les rideaux du salon.” |
| `samples/phrase3_qc.wav` | “La température va être de moins douze degrés demain matin, sors tes bottes.” |

## Repository contents

```
marie-line-fr-qc/
├── README.md
├── README.en.md
├── LICENSE
├── scripts/
│   ├── f5_batch_gen.py          # F5-TTS mass generation (infer_process loop)
│   ├── f5_build_dataset.sh      # build the Piper dataset (metadata.csv)
│   ├── train_qc_foreground.sh   # launch Piper training (foreground)
│   ├── export_voice_qc.sh       # ONNX + JSON export
│   ├── preprocess.sh            # Piper preprocessing (resampling, espeak)
│   ├── train.sh                 # Piper training script
│   └── f5_phrases_300.txt       # the 300 Quebec sentences used
├── samples/
│   ├── phrase1_qc.wav           # sample 1 (see table above)
│   ├── phrase2_qc.wav           # sample 2
│   └── phrase3_qc.wav           # sample 3
└── voices/
    ├── fr_FR-marie_line_fr_qc-medium.onnx
    └── fr_FR-marie_line_fr_qc-medium.onnx.json
```

## Usage in Home Assistant

1. Put `fr_FR-marie_line_fr_qc-medium.onnx` and `fr_FR-marie_line_fr_qc-medium.onnx.json` in `/share/piper/` (via SMB or the file editor).
2. **Restart the Piper add-on** (it rescans `/share/piper` on startup).
3. Use the voice through `tts.speak`:

```yaml
service: tts.speak
data:
  entity_id: media_player.my_speaker
  message: "Il fait vraiment beau aujourd'hui, prends une marche au bord du fleuve."
  options:
    voice: fr_FR-marie_line_fr_qc-medium
```

> **Known limitation**: custom voices do **not** show up in the add-on dropdown (static list); they remain usable via `options.voice`, which is enough for `tts.speak` and the voice assistant.
>
> If the voice sounds slow, adjust `length_scale` (e.g. 1.3) and `sentence_silence` (e.g. 0.5) in the add-on — independent of the model.

## Step-by-step recipe

### 1. Prerequisites

- NVIDIA GPU (12 GB VRAM is enough) with **Docker + NVIDIA Container Toolkit**, on WSL2.
- A “nightly” Piper image with CUDA (here `chatterbox-piper:nightly-cu128-sm120`, it contains `piper_train` and the `piper` runtime).
- F5-TTS **Quebec** model (`model_quebec_v1.safetensors` + `vocab.txt`) — see F5-TTS Quebec on Hugging Face (a **CC-BY-NC** model).
- Piper starting checkpoint: `fr_fr_siwis_medium` (`epoch=3304`, from the official Piper voices).

### 2. Reference file

From the reference voice file, extract a **short clip (~4.5 s)** representative of the target (tone and accent) and transcribe it exactly.

*Validated* generation settings:

```
short ref ≈ 4.5 s   (ref_court.wav)
nfe_step    = 64
cfg_strength = 3.0
```

### 3. Sentence list

Write ~300 varied sentences: home-automation commands, weather, imperative verbs, Quebec constructions (“pis”, “ben”, “tu” in questions), long and short sentences. See `f5_phrases_300.txt` for an example of the distribution.

### 4. F5-TTS mass generation

`f5_batch_gen.py` loops over the sentences with `f5_tts.infer.utils_infer.infer_process` (a single reference for all of them) and writes `qc_XXXX.wav` (24 kHz) plus a `manifest.txt` (`name|text`) that also acts as an **interruption resume point**.

Important pitfalls:
- The `f5_tts` CLI (`infer_cli.py`) **loads the model at module level**: do not `import f5_tts.infer.infer_cli`, import `f5_tts.infer.utils_infer`.
- `--gen_file` **concatenates all lines into a single audio**: useless for a batch of sentences.
- Disable Hugging Face network access with `HF_HUB_OFFLINE=1` (local cache) to avoid stalls during generation.

### 5. Piper dataset

`f5_build_dataset.sh`: copies the `qc_XXXX.wav` files into `wavs/` and builds `metadata.csv` in ljspeech format (`qc_0000.wav|text`) from the manifest.

### 6. Preprocessing

`preprocess.sh`: copies the dataset into a persistent named volume (once only — the 9p/WSL mount is fragile), normalizes and **resamples to 22.05 kHz** (sox), then runs `piper_train.preprocess` with espeak **fr** (ljspeech format, single speaker).

> F5 outputs are 24 kHz; Piper works at 22.05 kHz.

### 7. Fine-tuning

`train_qc_foreground.sh` mounts the dataset, the starting checkpoint (`/app/checkpoint.ckpt`) and working directories, then runs `preprocess.sh && train.sh`.

Settings used:

```
MAX_EPOCHS=3804   (= siwis epoch 3304 + 500)
QUALITY=medium
BATCH_SIZE=12
PRECISION=32
checkpoint-epochs=1
validation-split=0.0
```

- **Automatic resume**: `train.sh` resumes from the most recent checkpoint in `lightning_logs/version_*/` if one exists.
- The pipeline keeps only one checkpoint; expect ~830 MB per save.
- Duration: ~6 h on an RTX 3060 12 GB.

### 8. ONNX export

`export_voice_qc.sh`: selects the most recent checkpoint, exports it to ONNX via `piper_train.export_onnx` (with `PIPER_TORCH_ONNX_DYNAMO=0`), and copies `config.json` to `.onnx.json`.

Then fix two fields in the `.onnx.json` for HA deployment:

```json
"language": { "code": "fr_FR" },
"espeak":   { "voice": "fr" }
```

### 9. Deployment

See “Usage in Home Assistant” above.

## Lessons learned

- **Docker GPU containers + WSL**: a `docker run` (even with `-d`) dies with the WSL client session. Run training **in the foreground** in a session that stays open, log to a file, and be able to relaunch (“resumes from the last checkpoint”).
- **Quoting between PowerShell ↔ WSL ↔ bash**: always use **script files**; French apostrophes (“aujourd'hui”) cannot live inside a `bash -lc '...'` block.
- **espeak-ng handles French apostrophes correctly** (with a real apostrophe): `aujourd'hui` → `/oʒuʁdyi/`. Testing with a *space* (`aujourd hui`) yields a wrong pronunciation — always test with real apostrophes.
- Check the target pronunciation on a few sentences before running the big generation (a short multi-parameter PoC beats 300 files to redo).

## License

- **Scripts and documentation**: MIT (see `LICENSE`).
- **Voice `fr_FR-marie_line_fr_qc-medium`**: released under **CC-BY-NC-4.0** — non-commercial use, attribution required. It derives from the F5-TTS Quebec model (CC-BY-NC) and the `fr_fr_siwis_medium` checkpoint (Piper / rhasspy). Personal home-automation use is allowed.
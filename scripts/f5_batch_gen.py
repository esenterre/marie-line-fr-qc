import os

import soundfile as sf
from hydra.utils import get_class
from importlib.resources import files
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    device,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

MODEL = "F5TTS_Base"
CKPT_FILE = "/app/f5q/model_quebec_v1.safetensors"
VOCAB_FILE = "/app/f5q/vocab.txt"
REF_AUDIO = "/app/f5gen/ref_court.wav"
REF_TEXT = os.environ.get("REF_TEXT", "")
PHRASES_FILE = "/app/f5gen/phrases.txt"
OUT_DIR = "/app/f5gen/batch300"

NFE_STEP = 64
CFG_STRENGTH = 3.0

os.makedirs(OUT_DIR, exist_ok=True)

vocoder = load_vocoder(vocoder_name="vocos", device=device)
model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{MODEL}.yaml")))
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
ema_model = load_model(
    model_cls,
    model_cfg.model.arch,
    CKPT_FILE,
    mel_spec_type="vocos",
    vocab_file=VOCAB_FILE,
    device=device,
)

ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)

lines = [l.strip() for l in open(PHRASES_FILE, encoding="utf-8").read().splitlines() if l.strip()]

manifest = os.path.join(OUT_DIR, "manifest.txt")
done = set()
if os.path.exists(manifest):
    for row in open(manifest, encoding="utf-8"):
        if "|" in row:
            done.add(row.split("|", 1)[0])

with open(manifest, "a", encoding="utf-8") as mf:
    for i, text in enumerate(lines):
        name = f"qc_{i:04d}.wav"
        if name in done and os.path.exists(os.path.join(OUT_DIR, name)):
            print(f"skip {name}", flush=True)
            continue
        try:
            audio, sr, _ = infer_process(
                ref_audio,
                ref_text,
                text,
                ema_model,
                vocoder,
                mel_spec_type="vocos",
                nfe_step=NFE_STEP,
                cfg_strength=CFG_STRENGTH,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}", flush=True)
            continue
        sf.write(os.path.join(OUT_DIR, name), audio, sr)
        mf.write(f"{name}|{text}\n")
        mf.flush()
        print(f"done {name} ({i + 1}/{len(lines)})", flush=True)

print("BATCH DONE", flush=True)
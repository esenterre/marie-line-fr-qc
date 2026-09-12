[🇫🇷 Français](README.md) · [🇬🇧 English](README.en.md)

# Voix française à accent québécois pour Home Assistant (Piper)

Voix de synthèse **française à accent québécois** pour [Home Assistant](https://www.home-assistant.io/) via l'[add-on Piper](https://github.com/home-assistant/addons/tree/master/piper), réalisée pour **un usage personnel en domotique**.

Le point de départ était un **fichier de voix de référence** (une voix de synthèse) : le clip a servi à fine-tuner **F5-TTS Québec** (`model_quebec_v1`) pour générer ~300 phrases québécoises, qui ont ensuite servi de dataset pour **ré-entraîner Piper** (modèle VITS) à partir du checkpoint open-source `fr_fr_siwis_medium`. Le résultat final est assez éloigné de la source pour être considéré comme une voix distincte.

Ce dépôt contient la **recette complète**, les **scripts**, les **phrases**, des **extraits audio** et la **voix finale** (`.onnx` + `.onnx.json`). La référence audio, les 300 `.wav` générés et les checkpoints d'entraînement ne sont **pas** inclus.

## Écouter avant d'installer

Trois extraits générés avec la voix finale (dossier `samples/`) :

| Fichier | Texte |
|---|---|
| `samples/phrase1_qc.wav` | « Y fait vraiment beau aujourd'hui, on va jaser au bord du fleuve pis prendre un café. » |
| `samples/phrase2_qc.wav` | « Allume la lumière de la cuisine pis ferme les rideaux du salon. » |
| `samples/phrase3_qc.wav` | « La température va être de moins douze degrés demain matin, sors tes bottes. » |

## Fichiers livrés

```
marie-line-fr-qc/
├── README.md
├── README.en.md
├── LICENSE
├── scripts/
│   ├── f5_batch_gen.py          # génération de masse F5-TTS (boucle infer_process)
│   ├── f5_build_dataset.sh      # construction du dataset Piper (metadata.csv)
│   ├── train_qc_foreground.sh   # lancement de l'entraînement Piper (foreground)
│   ├── export_voice_qc.sh       # export ONNX + JSON
│   ├── preprocess.sh            # prétraitement Piper (rééchantillonnage, espeak)
│   ├── train.sh                 # script d'entraînement Piper
│   └── f5_phrases_300.txt       # les 300 phrases québécoises utilisées
├── samples/
│   ├── phrase1_qc.wav           # extrait 1 (voir tableau ci-dessus)
│   ├── phrase2_qc.wav           # extrait 2
│   └── phrase3_qc.wav           # extrait 3
└── voices/
    ├── fr_FR-marie_line_fr_qc-medium.onnx
    └── fr_FR-marie_line_fr_qc-medium.onnx.json
```

## Utilisation dans Home Assistant

1. Déposer `fr_FR-marie_line_fr_qc-medium.onnx` et `fr_FR-marie_line_fr_qc-medium.onnx.json` dans `/share/piper/` (via SMB ou file editor).
2. **Redémarrer l'add-on Piper** (il rescanne `/share/piper` au démarrage).
3. Utiliser la voix dans `tts.speak` :

```yaml
service: tts.speak
data:
  entity_id: media_player.ma_enceinte
  message: "Y fait vraiment beau aujourd'hui, prends une marche au bord du fleuve."
  options:
    voice: fr_FR-marie_line_fr_qc-medium
```

> **Limitation connue** : les voix custom n'apparaissent **pas** dans le menu déroulant de l'add-on (liste statique) ; elles restent utilisables via `options.voice`, ce qui suffit pour `tts.speak` et l'assistante vocale.
>
> Si la voix semble lente, régler `length_scale` (ex. 1.3) et `sentence_silence` (ex. 0.5) dans l'add-on — indépendant du modèle.

## Recette pas à pas

### 1. Prérequis

- GPU NVIDIA (12 Go VRAM suffisent) sous **Docker + NVIDIA Container Toolkit**, sur WSL2.
- Image Piper « nightly » avec CUDA (ici `chatterbox-piper:nightly-cu128-sm120`, contient `piper_train` et le runtime `piper`).
- Modèle F5-TTS **Québec** (`model_quebec_v1.safetensors` + `vocab.txt`) — voir F5-TTS Québec sur Hugging Face (modèle **CC-BY-NC**).
- Checkpoint de départ Piper : `fr_fr_siwis_medium` (`epoch=3304`, depuis les modèles Piper officiels).

### 2. Fichier de référence

Extraire du fichier de voix de référence un **clip court (~4,5 s)** représentatif (le timbre et l'accent cibles) et le transcrire exactement.

Configuration *validée* pour la génération :

```
réf courte ≈ 4,5 s    (ref_court.wav)
nfe_step    = 64
cfg_strength = 3.0
```

### 3. Phrases

Écrire une liste de ~300 phrases variées : commandes domotiques, météo, verbes à l'impératif, tournures québécoises (« pis », « ben », « tu » en interrogation), phrases longues et courtes. Voir `f5_phrases_300.txt` pour un exemple de distribution.

### 4. Génération de masse F5-TTS

`f5_batch_gen.py` boucle sur les phrases avec `f5_tts.infer.utils_infer.infer_process` (une seule référence pour toutes) et écrit `qc_XXXX.wav` (24 kHz) + un `manifest.txt` (`nom|texte`) qui sert aussi de **reprise après interruption**.

Pièges rencontrés (importants) :
- La CLI `f5_tts` (`infer_cli.py`) **charge le modèle au niveau module** : ne faites pas `import f5_tts.infer.infer_cli`, importez `f5_tts.infer.utils_infer`.
- `--gen_file` **concatène toutes les lignes en un seul audio** : inutilisable pour un lot de phrases.
- Couper le réseau Hugging Face avec `HF_HUB_OFFLINE=1` (cache local) pour éviter les blocages réseau pendant la génération.

### 5. Dataset Piper

`f5_build_dataset.sh` : copie des `qc_XXXX.wav` dans `wavs/` et génère `metadata.csv` au format ljspeech (`qc_0000.wav|texte`) depuis le manifest.

### 6. Prétraitement

`preprocess.sh` : copie du dataset dans un volume nommé persistant (une seule fois, jargon 9p/WSL fragile), normalisation + **rééchantillonnage 22,05 kHz** (sox), puis `piper_train.preprocess` avec espeak **fr** (format ljspeech, single-speaker).

> Les sorties F5 sont en 24 kHz ; Piper travaille en 22,05 kHz.

### 7. Fine-tune

`train_qc_foreground.sh` monte le dataset, le checkpoint de départ (`/app/checkpoint.ckpt`) et des dossiers de travail, puis enchaîne `preprocess.sh && train.sh`.

Paramètres utilisés :

```
MAX_EPOCHS=3804   (= epoch 3304 du siwis + 500)
QUALITY=medium
BATCH_SIZE=12
PRECISION=32
checkpoint-epochs=1
validation-split=0.0
```

- **Reprise automatique** : `train.sh` reprend le checkpoint le plus récent de `lightning_logs/version_*/` s'il existe.
- Le pipeline ne conserve qu'un checkpoint ; prévoir ~830 Mo par sauvegarde.
- Durée : ~6 h sur RTX 3060 12 Go.

### 8. Export ONNX

`export_voice_qc.sh` : sélectionne le checkpoint le plus récent, l'expororte en ONNX via `piper_train.export_onnx` (avec `PIPER_TORCH_ONNX_DYNAMO=0`), copie le `config.json` en `.onnx.json`.

Corriger ensuite deux champs du `.onnx.json` pour le déploiement HA :

```json
"language": { "code": "fr_FR" },
"espeak":   { "voice": "fr" }
```

### 9. Déploiement

Voir « Utilisation dans Home Assistant » ci-dessus.

## Leçons apprises

- **Conteneurs Docker GPU + WSL** : un `docker run` (même `-d`) meurt avec la session WSL du client. Lancer l'entraînement **en foreground** dans une session qui reste ouverte, avec un log vers un fichier et une relance possible (« reprend au dernier checkpoint »).
- **Quoting entre PowerShell ↔ WSL ↔ bash** : toujours passer par des **fichiers de scripts** ; les apostrophes françaises (« aujourd'hui ») ne supportent pas d'être dans un bloc `bash -lc '...'`.
- **espeak-ng gère bien les apostrophes françaises** (avec une vraie apostrophe) : `aujourd'hui` → `/oʒuʁdyi/`. Un test avec un *espace* (`aujourd hui`) donne une prononciation erronée — testez toujours avec de vraies apostrophes.
- Vérifier la prononciation cible sur quelques phrases avant de lancer la grosse génération (un court PoC à plusieurs paramètres vaut mieux que 300 fichiers à refaire).

## Licence

- **Scripts et documentation** : MIT (voir `LICENSE`).
- **Voix `fr_FR-marie_line_fr_qc-medium`** : publiée sous **CC-BY-NC-4.0** — usage non commercial, attribution requise. Elle dérive du modèle F5-TTS Québec (CC-BY-NC) et du checkpoint `fr_fr_siwis_medium` (Piper / rhasspy). Utilisation personnelle en domotique autorisée.
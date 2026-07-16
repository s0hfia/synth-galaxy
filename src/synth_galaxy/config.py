from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PRESETS_DIR = PROJECT_ROOT / "presets"
AUDIO_DIR = DATA_DIR / "audio"
PATCHES_DIR = DATA_DIR / "patches"
FEATURES_DIR = DATA_DIR / "features"

VITAL_VST3_PATH = Path("/Library/Audio/Plug-Ins/VST3/Vital.vst3")

SAMPLE_RATE = 44100
RENDER_SECONDS = 2.0
RENDER_NOTE = 60  # middle C
RENDER_VELOCITY = 100
NOTE_HOLD_SECONDS = 1.5  # release at 1.5s, let tail ring through 2.0s

for d in (DATA_DIR, PRESETS_DIR, AUDIO_DIR, PATCHES_DIR, FEATURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

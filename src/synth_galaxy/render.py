from pathlib import Path

import dawdreamer as daw
import numpy as np
import soundfile as sf

from .config import (
    NOTE_HOLD_SECONDS,
    RENDER_NOTE,
    RENDER_SECONDS,
    RENDER_VELOCITY,
    SAMPLE_RATE,
    VITAL_VST3_PATH,
)


def make_engine(sample_rate: int = SAMPLE_RATE, block_size: int = 512) -> daw.RenderEngine:
    return daw.RenderEngine(sample_rate=sample_rate, block_size=block_size)


def load_vital(engine: daw.RenderEngine, vst3_path: Path = VITAL_VST3_PATH):
    if not vst3_path.exists():
        raise FileNotFoundError(f"Vital.vst3 not found at {vst3_path}")
    return engine.make_plugin_processor("vital", str(vst3_path))


def render_note(
    engine: daw.RenderEngine,
    synth,
    note: int = RENDER_NOTE,
    velocity: int = RENDER_VELOCITY,
    duration: float = RENDER_SECONDS,
    hold: float = NOTE_HOLD_SECONDS,
) -> np.ndarray:
    synth.clear_midi()
    synth.add_midi_note(note, velocity, 0.0, hold)
    engine.load_graph([(synth, [])])
    engine.render(duration)
    return engine.get_audio()


def write_wav(out_path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dawdreamer returns (channels, samples); soundfile wants (samples, channels)
    sf.write(out_path, audio.T, sample_rate)

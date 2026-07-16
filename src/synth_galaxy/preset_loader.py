"""Flat-params preset loader.

Maps the snake_case keys in a .vital JSON's `settings` dict to dawdreamer's
host-exposed parameter indices, then drives those via `set_parameter`.

Limitations (lifted by Path Y when complete):
- Wavetables (oscillators play Vital's default sine wave instead)
- Custom LFO shapes (only LFO frequency/sync/etc. are set, not the drawn curves)
- Modulation routing topology (source/destination pairs are not host-exposed,
  but routing amounts ARE settable through "Modulation N Amount" host params)
- Sample-track audio data
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


# JSON keys in .vital use abbreviations; host params spell them out.
# Discovered empirically by comparing JSON keys to host param names.
JSON_TO_HOST_ABBREVIATIONS = {
    "osc": "oscillator",
    "env": "envelope",
    "mod_wheel": "mod wheel",
    "pitch_wheel": "pitch wheel",
}


def snake_to_title(snake: str) -> str:
    return snake.replace("_", " ").title()


def candidate_host_names(json_key: str) -> list[str]:
    """Generate likely host parameter names for a given JSON key."""
    cands = [snake_to_title(json_key)]
    for short, long in JSON_TO_HOST_ABBREVIATIONS.items():
        prefix = f"{short}_"
        if json_key.startswith(prefix) or f"_{prefix}" in json_key:
            expanded = json_key.replace(prefix, f"{long.replace(' ', '_')}_", 1)
            cands.append(snake_to_title(expanded))
    return cands


def build_mapping(synth, json_keys: list[str]) -> tuple[dict[str, int], list[str]]:
    """Match JSON keys to host param indices. Returns (mapping, unmatched_keys)."""
    n = synth.get_plugin_parameter_size()
    host_by_name: dict[str, int] = {
        synth.get_parameter_name(i): i for i in range(n)
    }
    mapping: dict[str, int] = {}
    unmatched: list[str] = []
    for key in json_keys:
        for cand in candidate_host_names(key):
            if cand in host_by_name:
                mapping[key] = host_by_name[cand]
                break
        else:
            unmatched.append(key)
    return mapping, unmatched


def load_vital_json(path: Path) -> dict:
    """Parse a .vital file. Handles plain JSON and gzipped JSON."""
    data = path.read_bytes()
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return json.loads(gzip.decompress(data))


def flat_scalar_settings(settings: dict) -> dict[str, float]:
    """Extract only the scalar float/int/bool entries from a settings dict."""
    return {
        k: float(v)
        for k, v in settings.items()
        if isinstance(v, (int, float, bool)) and not isinstance(v, str)
    }


def apply_preset_to_synth(
    synth,
    preset_settings: dict,
    mapping: dict[str, int],
    defaults: dict[int, float],
) -> int:
    """Reset to defaults, then apply preset values via name mapping.

    Returns the number of params actually written.
    """
    for idx, val in defaults.items():
        synth.set_parameter(idx, val)
    flat = flat_scalar_settings(preset_settings)
    written = 0
    for key, idx in mapping.items():
        if key in flat:
            synth.set_parameter(idx, flat[key])
            written += 1
    return written

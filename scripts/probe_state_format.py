"""Decode the VC2! state wrapper and see if .vital JSON is hiding inside."""

import base64
import gzip
import re
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

from synth_galaxy.config import SAMPLE_RATE
from synth_galaxy.render import load_vital, make_engine, render_note


PRESET = Path("/Users/sof/Music/Vital/Organisms/BS Cyclops.vital")


def render_and_stat(synth, engine, label: str):
    audio = render_note(engine, synth)
    rms = float(np.sqrt((audio ** 2).mean()))
    peak = float(np.abs(audio).max())
    print(f"  [{label}] shape={audio.shape}  rms={rms:.4f}  peak={peak:.4f}")
    out = Path(__file__).parent.parent / "data" / "audio" / f"state_test_{label}.wav"
    sf.write(out, audio.T, SAMPLE_RATE)


def main() -> None:
    engine = make_engine()
    synth = load_vital(engine)

    # Capture baseline render
    print("--- baseline (default state) ---")
    render_and_stat(synth, engine, "00_default")

    # Try loading the .vital file directly via load_state(filepath)
    print("\n--- attempt 1: load_state(.vital path) ---")
    try:
        result = synth.load_state(str(PRESET))
        print(f"  load_state returned: {result!r}")
        render_and_stat(synth, engine, "01_load_state_vital")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Try load_vst3_preset
    print("\n--- attempt 2: load_vst3_preset(.vital path) ---")
    try:
        result = synth.load_vst3_preset(str(PRESET))
        print(f"  load_vst3_preset returned: {result!r}")
        render_and_stat(synth, engine, "02_load_vst3")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Now actually decode what save_state wrote, to understand the wrapper
    print("\n--- decoding the VC2! wrapper ---")
    state_path = Path(__file__).parent.parent / "data" / "default_state.bin"
    raw = state_path.read_bytes()
    print(f"raw size: {len(raw):,}")
    assert raw[:4] == b"VC2!", "expected VC2! magic"
    xml_len = int.from_bytes(raw[4:8], "little")
    print(f"declared XML length: {xml_len:,}")
    xml = raw[8 : 8 + xml_len].decode("utf-8", errors="replace")
    print(f"actual XML extracted: {len(xml):,} chars")
    # Find IComponent base64 content
    m = re.search(r"<IComponent>(.*?)</IComponent>", xml, re.DOTALL)
    if not m:
        print("  no <IComponent> found")
        return
    b64 = m.group(1).strip()
    print(f"IComponent base64 length: {len(b64):,}")
    # The base64 alphabet here uses '.' instead of '=' for padding? Let's see what chars are present
    chars = set(b64)
    print(f"  distinct chars in b64 sample (first 30): {sorted(chars)[:30]}")

    # JUCE's MemoryBlock::toBase64Encoding uses a custom alphabet:
    # digits 0-9 then A-Z then a-z then '.' and other chars. Let me try fromBase64
    # Actually JUCE's Base64::toBase64 uses standard alphabet, but MemoryBlock has its own.
    # Try standard b64 first replacing '.' with '='
    decoded = None
    for variant in ["standard", "dot_to_eq", "juce_memoryblock"]:
        try:
            if variant == "standard":
                decoded = base64.b64decode(b64)
            elif variant == "dot_to_eq":
                decoded = base64.b64decode(b64.replace(".", "="))
            elif variant == "juce_memoryblock":
                # JUCE MemoryBlock format: starts with a varint length, then base64 of the data
                # Skip leading digits + '.' separator, then standard base64
                m2 = re.match(r"^(\d+)\.(.*)$", b64, re.DOTALL)
                if m2:
                    declared = int(m2.group(1))
                    payload = m2.group(2)
                    print(f"  juce_memoryblock: declared decoded len={declared}, payload len={len(payload)}")
                    decoded = base64.b64decode(payload + "=" * (-len(payload) % 4))
                else:
                    continue
            print(f"  [{variant}] decoded {len(decoded):,} bytes")
            print(f"  first 80 bytes hex: {decoded[:80].hex()}")
            print(f"  first 80 ascii: {decoded[:80]!r}")
            # Test if it's gzip
            if decoded[:2] == b"\x1f\x8b":
                try:
                    inflated = gzip.decompress(decoded)
                    print(f"  GZIP -> {len(inflated):,} bytes")
                    print(f"  first 200: {inflated[:200]!r}")
                except Exception as e:
                    print(f"  gzip failed: {e}")
            # Test if it's zlib
            try:
                inflated = zlib.decompress(decoded)
                print(f"  ZLIB -> {len(inflated):,} bytes")
                print(f"  first 200: {inflated[:200]!r}")
            except Exception:
                pass
            # Test if raw JSON
            try:
                start = decoded.find(b"{")
                if start >= 0:
                    print(f"  raw json '{{' at offset {start}; snippet: {decoded[start:start+200]!r}")
            except Exception:
                pass
            break
        except Exception as e:
            print(f"  [{variant}] failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

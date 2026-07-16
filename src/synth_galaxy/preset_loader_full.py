"""Full-fidelity Vital preset loader.

Loads complete `.vital` JSON presets (including wavetables, LFO shapes,
modulation routings, sample data) into a dawdreamer-hosted Vital.vst3 via
`synth.load_state(<path>)`.

Why this is non-trivial
-----------------------
dawdreamer's `load_preset()` does not understand `.vital`'s JSON format,
and `load_state()` requires the *exact* binary format that JUCE's VST3 host
wrapper produces — not the raw `.vital` JSON. We reverse-engineered the
wrapper by reading JUCE source:

State file structure (what `synth.save_state` writes / `synth.load_state` reads):

    [0..4)   uint32 LE  magic       = 0x21324356  (ASCII "VC2!")
    [4..8)   uint32 LE  xml_length  = total file size - 9
    [8..8+L) bytes      XML payload (UTF-8, no trailing newline)
    [8+L]    byte       0x00 null terminator

The XML payload looks like:

    <?xml version="1.0" encoding="UTF-8"?>
    <VST3PluginState>
      <IComponent>LEN.JUCE_BASE64_DATA</IComponent>
      <IEditController>LEN.JUCE_BASE64_DATA</IEditController>
    </VST3PluginState>

Both base64 blobs use JUCE's custom `MemoryBlock::toBase64Encoding()` format:

    "<decimal_length>.<custom_base64>"

with the alphabet
".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
(NOT standard base64 — '.' is the first symbol, '+' is 63, no '/' or '=').
No padding; characters encode 6-bit values that are written into a bit
stream of exactly `length * 8` bits.

The decoded `IComponent` bytes are what JUCE's VST3 wrapper passes to
Vital's `setStateInformation(const void*, int)`. Vital reads those bytes
as a single UTF-8 string (via JUCE's `MemoryInputStream::readEntireStream
AsString`) and `json::parse`s them. So all we need to write into the
IComponent slot is the raw JSON text of the `.vital` file plus a single
trailing null byte (matching what Vital's `getStateInformation` produces
via `MemoryOutputStream::writeString`).

`IEditController` we copy verbatim from a captured default-state dump.
Vital's edit controller state is parameter-mirror state; the IComponent
load is what drives the actual audio engine.

References
----------
- mtytel/vital src/plugin/synth_plugin.cpp : SynthPlugin::getStateInformation
- juce-framework/JUCE juce_audio_processors/processors/juce_AudioProcessor.cpp :
  AudioProcessor::copyXmlToBinary / getXmlFromBinary
  uses magic 0x21324356 + 4-byte length + XML + null
- juce-framework/JUCE juce_audio_processors/format_types/juce_VST3PluginFormat.cpp :
  VST3PluginInstance::getStateInformation calls appendStateFrom which
  base64-encodes via MemoryBlock::toBase64Encoding
- juce-framework/JUCE juce_core/memory/juce_MemoryBlock.cpp :
  toBase64Encoding / fromBase64Encoding implementation
"""

from __future__ import annotations

import io
import json
import re
import struct
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# JUCE MemoryBlock base64 (custom alphabet, NOT standard RFC 4648)
# ---------------------------------------------------------------------------
# JUCE writes len + '.' + chars where each char encodes 6 bits of payload.
# The bits are packed LSB-first into a flat bit stream of length numBytes*8.
# Trailing bits in the final 6-bit group are zero. So the number of
# characters is ceil(numBytes * 8 / 6).

_JUCE_B64_ALPHABET = (
    ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
)
assert len(_JUCE_B64_ALPHABET) == 64

_JUCE_B64_DECODE = {c: i for i, c in enumerate(_JUCE_B64_ALPHABET)}


def juce_base64_decode(s: str) -> bytes:
    """Decode JUCE MemoryBlock::toBase64Encoding string back to bytes.

    Format: "<decimal_length>.<custom_base64_chars>"
    """
    m = re.match(r"^(\d+)\.(.*)$", s, re.DOTALL)
    if not m:
        raise ValueError(f"not a JUCE MemoryBlock base64 string: {s[:60]!r}")
    num_bytes = int(m.group(1))
    payload = m.group(2)

    # Bit-stream reconstruction. Each char writes 6 bits at offset i*6
    # using setBitRange semantics (LSB-first within each byte).
    out = bytearray(num_bytes)
    for i, ch in enumerate(payload):
        try:
            val = _JUCE_B64_DECODE[ch]
        except KeyError:
            raise ValueError(
                f"invalid JUCE base64 char {ch!r} at index {i}"
            ) from None
        bit_offset = i * 6
        for b in range(6):
            if val & (1 << b):
                byte_idx = (bit_offset + b) >> 3
                bit_idx = (bit_offset + b) & 7
                if byte_idx < num_bytes:
                    out[byte_idx] |= 1 << bit_idx
    return bytes(out)


def juce_base64_encode(data: bytes) -> str:
    """Encode bytes to JUCE MemoryBlock::toBase64Encoding format.

    Inverse of `juce_base64_decode`. The output is "<len>.<chars>" where each
    char represents 6 bits read from the source bit stream (LSB-first within
    each byte), padded with zeros at the end.
    """
    num_bytes = len(data)
    num_bits = num_bytes * 8
    # Number of 6-bit groups needed to cover every bit (round up).
    num_chars = (num_bits + 5) // 6

    chars: list[str] = []
    for i in range(num_chars):
        bit_offset = i * 6
        val = 0
        for b in range(6):
            bo = bit_offset + b
            if bo >= num_bits:
                break
            byte_idx = bo >> 3
            bit_idx = bo & 7
            if data[byte_idx] & (1 << bit_idx):
                val |= 1 << b
        chars.append(_JUCE_B64_ALPHABET[val])
    return f"{num_bytes}.{''.join(chars)}"


# ---------------------------------------------------------------------------
# VC2!-framed state file (AudioProcessor::copyXmlToBinary format)
# ---------------------------------------------------------------------------

_VC2_MAGIC = 0x21324356  # little-endian == b"VC2!"


def parse_state_file(raw: bytes) -> dict:
    """Parse a VC2! state file into its constituent parts.

    Returns dict with keys:
      magic_ok: bool
      xml_length: int  (declared)
      xml: str         (the XML text)
      icomponent_b64: str   (raw JUCE base64 from IComponent element)
      ieditcontroller_b64: str | None
      icomponent_bytes: bytes  (decoded IComponent payload — usually a JSON string + null)
      icomponent_json: dict | None  (parsed Vital JSON if decodable)
    """
    if len(raw) < 8:
        raise ValueError(f"state file too short: {len(raw)} bytes")
    magic = struct.unpack("<I", raw[:4])[0]
    xml_length = struct.unpack("<I", raw[4:8])[0]
    magic_ok = magic == _VC2_MAGIC
    if not magic_ok:
        raise ValueError(
            f"bad magic: got 0x{magic:08x}, expected 0x{_VC2_MAGIC:08x}"
        )

    xml = raw[8 : 8 + xml_length].decode("utf-8", errors="replace")

    icomponent_b64 = ""
    ieditcontroller_b64 = None
    m = re.search(r"<IComponent>(.*?)</IComponent>", xml, re.DOTALL)
    if m:
        icomponent_b64 = m.group(1).strip()
    m = re.search(r"<IEditController>(.*?)</IEditController>", xml, re.DOTALL)
    if m:
        ieditcontroller_b64 = m.group(1).strip()

    icomponent_bytes = (
        juce_base64_decode(icomponent_b64) if icomponent_b64 else b""
    )

    icomponent_json: Optional[dict] = None
    if icomponent_bytes:
        # Vital writes JSON text + trailing null byte. Strip trailing nulls.
        text = icomponent_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        # Trim to the JSON object portion (start at first '{').
        start = text.find("{")
        if start >= 0:
            try:
                icomponent_json = json.loads(text[start:])
            except json.JSONDecodeError:
                pass

    return {
        "magic_ok": magic_ok,
        "xml_length": xml_length,
        "xml": xml,
        "icomponent_b64": icomponent_b64,
        "ieditcontroller_b64": ieditcontroller_b64,
        "icomponent_bytes": icomponent_bytes,
        "icomponent_json": icomponent_json,
    }


def _build_xml(icomponent_b64: str, ieditcontroller_b64: Optional[str]) -> str:
    """Build the VST3PluginState XML matching JUCE's single-line format.

    JUCE's `XmlElement::writeTo` with `TextFormat().singleLine()` produces:
      <?xml version="1.0" encoding="UTF-8"?> <VST3PluginState>...

    Note the single space between header and root element (because
    singleLine() sets newLineChars=nullptr, and JUCE writes a literal space
    instead of newlines after the XML decl in that mode). Elements are
    concatenated with no inter-element whitespace.
    """
    parts = ['<?xml version="1.0" encoding="UTF-8"?> <VST3PluginState>']
    parts.append(f"<IComponent>{icomponent_b64}</IComponent>")
    if ieditcontroller_b64 is not None:
        parts.append(
            f"<IEditController>{ieditcontroller_b64}</IEditController>"
        )
    parts.append("</VST3PluginState>")
    return "".join(parts)


def build_state_file(
    icomponent_bytes: bytes,
    ieditcontroller_b64: Optional[str] = None,
) -> bytes:
    """Wrap raw IComponent bytes in the full VC2! state-file frame.

    Parameters
    ----------
    icomponent_bytes : bytes
        Payload to deliver to the plugin's `setStateInformation`. For Vital
        this should be the UTF-8 JSON of the `.vital` preset followed by
        a single null byte (matching what Vital's own getStateInformation
        produces).
    ieditcontroller_b64 : str | None
        Optional pre-encoded JUCE-base64 string for the IEditController
        element. If None, the element is omitted; Vital tolerates that
        (the audio-engine state lives entirely in IComponent).
    """
    icomp_b64 = juce_base64_encode(icomponent_bytes)
    xml = _build_xml(icomp_b64, ieditcontroller_b64)
    xml_bytes = xml.encode("utf-8")
    xml_len = len(xml_bytes)

    buf = io.BytesIO()
    buf.write(struct.pack("<I", _VC2_MAGIC))
    buf.write(struct.pack("<I", xml_len))
    buf.write(xml_bytes)
    buf.write(b"\x00")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Cached default IEditController string, loaded lazily from data/default_state.bin.
_DEFAULT_IEDITCONTROLLER_B64: Optional[str] = None


def _get_default_ieditcontroller_b64() -> Optional[str]:
    """Pull IEditController bytes from the captured default_state.bin.

    Including IEditController in the wrapped state isn't strictly required
    (Vital's audio engine reads only IComponent), but matching JUCE's full
    two-element XML keeps us closer to the real save format and avoids any
    edge case where the host insists on both being present.
    """
    global _DEFAULT_IEDITCONTROLLER_B64
    if _DEFAULT_IEDITCONTROLLER_B64 is not None:
        return _DEFAULT_IEDITCONTROLLER_B64
    # Find the project's default_state.bin relative to this module.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "data" / "default_state.bin",  # project_root/data/
        here.parent / "data" / "default_state.bin",
    ]
    for p in candidates:
        if p.exists():
            parsed = parse_state_file(p.read_bytes())
            _DEFAULT_IEDITCONTROLLER_B64 = parsed["ieditcontroller_b64"]
            return _DEFAULT_IEDITCONTROLLER_B64
    return None


def vital_json_to_state_file(vital_json: dict, out_path: Path) -> None:
    """Serialize a parsed `.vital` JSON dict to a file that
    `dawdreamer`'s `synth.load_state(<path>)` will actually load.

    Parameters
    ----------
    vital_json : dict
        The parsed contents of a `.vital` preset (plain JSON).
    out_path : Path
        Where to write the wrapped state file (any extension is fine;
        `.bin` matches the convention used by `synth.save_state`).
    """
    out_path = Path(out_path)

    # Emit JSON with the same separators Vital's nlohmann::json::dump() uses
    # (no whitespace by default). The fidelity of the dump matters because
    # Vital re-parses this string verbatim — and floating-point round-trips
    # need full precision so wavetable / modulation values land exactly.
    json_text = json.dumps(vital_json, separators=(",", ":"), ensure_ascii=False)

    # Vital writes `MemoryOutputStream::writeString(...)` which appends a
    # single null byte after the UTF-8 payload. Reproduce that.
    icomponent_bytes = json_text.encode("utf-8") + b"\x00"

    iedit_b64 = _get_default_ieditcontroller_b64()
    state_bytes = build_state_file(icomponent_bytes, iedit_b64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(state_bytes)


def vital_file_to_state_file(vital_path: Path, out_path: Path) -> None:
    """Convenience wrapper: read a `.vital` file from disk and convert it.

    Mirrors `vital_json_to_state_file` but takes a path to a `.vital` file.
    """
    vital_path = Path(vital_path)
    data = json.loads(vital_path.read_text(encoding="utf-8"))
    vital_json_to_state_file(data, out_path)

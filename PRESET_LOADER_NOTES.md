# Preset loader: the state-file format that `synth.load_state` actually wants

`dawdreamer.PluginProcessor.load_state(path)` does **not** read `.vital` files
directly — it reads the bytes back into JUCE's `AudioPluginInstance::set
StateInformation`, which for a VST3 plugin is implemented by JUCE's host-side
VST3 wrapper. That wrapper expects a very specific binary frame. Here it is.

## Outer frame (8-byte header + XML + null)

This is exactly what `AudioProcessor::copyXmlToBinary` produces in JUCE:

```
offset  size  field
------  ----  -----
 0      4     magic        = 0x21324356  (little-endian; bytes spell "VC2!")
 4      4     xml_length   = total_file_size - 9
 8      L     xml_bytes    UTF-8, JUCE single-line format with XML decl
 8+L    1     null         0x00
```

JUCE source: `modules/juce_audio_processors/processors/juce_AudioProcessor.cpp`,
function `copyXmlToBinary`. The magic constant is named `magicXmlNumber`.

## XML payload

JUCE writes with `XmlElement::TextFormat().singleLine()`, which keeps the
default header (`addDefaultHeader=true`) and a `\r\n` after the declaration.
Everything else is one long line, no inter-element whitespace:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<VST3PluginState><IComponent>LEN.JUCE_BASE64</IComponent><IEditController>LEN.JUCE_BASE64</IEditController></VST3PluginState>
```

JUCE source: `juce_VST3PluginFormat.cpp`, `VST3PluginInstance::getState
Information` calls `appendStateFrom` for each of `IComponent` and
`IEditController`. Each child element wraps the plugin's own state bytes in
`MemoryBlock::toBase64Encoding()`.

## JUCE's custom base64 (NOT RFC 4648)

`MemoryBlock::toBase64Encoding()` produces strings like

```
174925.VMGcWAibPb...
```

i.e. `<decimal_length>.<custom_base64>`. Properties:

- The leading number is the **decoded byte length** in decimal.
- Alphabet (6-bit values 0..63):
  `".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"`
  Note `'.'` is *position 0* of the alphabet (not a padding sentinel!) and
  `'+'` is 63. There's no `/` and no `=` padding.
- Bits are packed **LSB-first within each byte**. Char `i` encodes bits
  `[i*6, i*6+6)` of the flat bit stream of length `num_bytes * 8`. Trailing
  bits in the final 6-bit group are zero.
- Number of chars = `ceil(num_bytes * 8 / 6)`.

JUCE source: `modules/juce_core/memory/juce_MemoryBlock.cpp`.

## The IComponent payload (Vital-specific)

What goes inside `<IComponent>...</IComponent>` after base64-decoding is the
exact byte stream that JUCE's VST3 plugin-side wrapper hands to Vital's
`AudioProcessor::setStateInformation(const void*, int)`. Vital's
implementation:

```cpp
// src/plugin/synth_plugin.cpp
void SynthPlugin::setStateInformation(const void* data, int size_in_bytes) {
  MemoryInputStream stream(data, size_in_bytes, false);
  String data_string = stream.readEntireStreamAsString();
  json json_data = json::parse(data_string.toStdString());
  LoadSave::jsonToState(this, save_info_, json_data);
  if (json_data.count("tuning"))
    getTuning()->jsonToState(json_data["tuning"]);
}
```

So the IComponent bytes are **just the `.vital` JSON text**. Vital's matching
`getStateInformation` writes it via `MemoryOutputStream::writeString`, which
in JUCE appends a single null terminator to the UTF-8 string. We match that
exactly: encode `json.dumps(...)` UTF-8 + `b"\x00"`.

`IEditController` is the parameter-mirror state. Including it isn't strictly
necessary for audio rendering — Vital reads the engine state out of
`IComponent` — but we include a verbatim copy of the bytes captured from a
fresh `synth.save_state()` of default Vital, so the file matches the shape
the host expects.

## Reverse-engineering trail

1. Read `mtytel/vital` `src/plugin/synth_plugin.cpp` to learn that `getState
   Information` writes plain JSON via `MemoryOutputStream::writeString`. No
   gzip, no extra wrapper inside Vital itself.
2. Read JUCE `juce_audio_processors/format_types/juce_VST3PluginFormat.cpp`
   to learn that the host-side wrapper produces an XML with `<IComponent>`
   and `<IEditController>` base64 children, then runs it through `copyXml
   ToBinary`.
3. Read JUCE `juce_audio_processors/processors/juce_AudioProcessor.cpp`
   `copyXmlToBinary` to learn about the `0x21324356` magic + length-prefix +
   `\x00` terminator. Magic in little-endian = ASCII "VC2!".
4. Read JUCE `juce_core/memory/juce_MemoryBlock.cpp` `toBase64Encoding` /
   `fromBase64Encoding` for the custom alphabet and bit packing.

`scripts/test_full_preset_load.py` cross-checks step 4 by re-encoding the
captured `data/default_state.bin` IComponent payload and verifying our
encoder reproduces JUCE's bytes verbatim.

## API

```python
from synth_galaxy.preset_loader_full import vital_json_to_state_file
import json
from pathlib import Path

vital_data = json.loads(Path("BS Cyclops.vital").read_text())
state_path = Path("/tmp/cyclops.state.bin")
vital_json_to_state_file(vital_data, state_path)

# Now this actually loads everything — wavetables, LFO shapes, modulations:
synth.load_state(str(state_path))
```

## Gotchas

- **Use `json.dumps(..., separators=(",", ":"))`.** Vital re-parses the text
  byte-for-byte; matching nlohmann's compact dump format keeps things small
  and predictable. Pretty-printing would still work because JSON parsing is
  whitespace-tolerant, but it inflates the file 3-5x.
- **Keep the trailing null byte.** Without it, `MemoryInputStream::read
  EntireStreamAsString` will still read the JSON, but it costs nothing to
  match what Vital actually writes — and it's robust against any code path
  that does treat the buffer as a C string.
- **Don't standard-base64 the IComponent.** JUCE's MemoryBlock format is
  custom; standard base64 will silently decode to garbage and `load_state`
  will quietly no-op (Vital's `json::parse` throws and is swallowed).
- **`load_state` returns `None` on success.** It only returns `False` if the
  file doesn't exist. Errors inside `setStateInformation` are swallowed by
  Vital's own try/except — silent no-op is the failure mode.

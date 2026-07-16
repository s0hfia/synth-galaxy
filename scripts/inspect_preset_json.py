"""Deep-dive into .vital JSON: what's stored, what we'd lose if we only set flat params."""

import json
from pathlib import Path

PRESETS = [
    Path("/Users/sof/Music/Vital/Organisms/BS Cyclops.vital"),
    Path("/Users/sof/Music/Vital/Factory").glob("**/*.vital"),
]


def inspect(path: Path) -> None:
    print(f"\n{'='*70}\n{path.name}  ({path.stat().st_size:,} bytes)")
    data = json.loads(path.read_bytes())
    print(f"Top-level keys: {list(data.keys())}")

    settings = data.get("settings", {})
    print(f"\nsettings: {len(settings)} entries")

    # Classify settings entries by value type
    types = {}
    for k, v in settings.items():
        t = type(v).__name__
        types.setdefault(t, []).append(k)

    for t, ks in sorted(types.items()):
        print(f"  type {t}: {len(ks)} entries")
        sample = ks[:5]
        print(f"    sample: {sample}")

    # Look for the big nested structures
    for k, v in settings.items():
        if isinstance(v, (list, dict)) and not isinstance(v, str):
            size = len(v)
            preview = ""
            if isinstance(v, list) and v:
                preview = f"first item: {type(v[0]).__name__}"
                if isinstance(v[0], dict):
                    preview += f", keys={list(v[0].keys())[:8]}"
            elif isinstance(v, dict) and v:
                preview = f"keys: {list(v.keys())[:8]}"
            print(f"\n  COMPLEX  '{k}' (size={size}): {preview}")


def main() -> None:
    # First check BS Cyclops
    inspect(PRESETS[0])
    # Try one Factory preset for comparison
    factory = list(Path("/Users/sof/Music/Vital/Factory").rglob("*.vital"))
    if factory:
        inspect(factory[0])


if __name__ == "__main__":
    main()

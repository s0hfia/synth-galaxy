"""Dump Vital's parameter space to a clean JSON for curation.

VST3 host params are normalized to [0,1], so we just need idx -> name and current default.
Also groups by name prefix so we can see categories at a glance.
"""

import json
from collections import Counter, defaultdict

from synth_galaxy.config import DATA_DIR
from synth_galaxy.render import load_vital, make_engine


def main() -> None:
    engine = make_engine()
    synth = load_vital(engine)

    n = synth.get_plugin_parameter_size()
    print(f"Total parameter count: {n}")

    params = []
    prefix_counter: Counter[str] = Counter()
    by_prefix: dict[str, list[dict]] = defaultdict(list)

    for i in range(n):
        name = synth.get_parameter_name(i)
        val = synth.get_parameter(i)
        entry = {"idx": i, "name": name, "default": float(val)}
        params.append(entry)

        # Group by first word of name as a rough category
        prefix = name.split()[0] if name else "<empty>"
        prefix_counter[prefix] += 1
        by_prefix[prefix].append(entry)

    out = DATA_DIR / "vital_params.json"
    out.write_text(json.dumps(params, indent=2))
    print(f"Wrote {n} params to {out}")

    print("\nTop 30 name prefixes (rough categories):")
    for prefix, count in prefix_counter.most_common(30):
        print(f"  {count:5d}  {prefix}")

    print("\nSample names from a few top categories:")
    for prefix, _ in prefix_counter.most_common(8):
        sample = by_prefix[prefix][:6]
        print(f"\n  [{prefix}] (showing 6 of {prefix_counter[prefix]}):")
        for p in sample:
            print(f"    {p['idx']:4d}  {p['name']}")


if __name__ == "__main__":
    main()

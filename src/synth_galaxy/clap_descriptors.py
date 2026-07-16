"""Curated semantic descriptor vocabulary for CLAP-based labeling.

Each descriptor has a display label (shown in the UI) and a query string
(the natural-language phrase encoded by CLAP). Phrasing matters — CLAP was
trained on captions, so 'a sound that is bright and metallic' generally
works better than the bare adjective 'bright'.

Sources for the vocabulary:
- Reymore & Huron 2020: 20-dimensional model of musical instrument timbre
  (rumbling, soft, watery, nasal, full, shrill, ringing, percussive,
   pure, brassy/metallic, raspy, sparkling, airy, resonant, hollow,
   woody, focused, muted, sustained).
- AudioCommons Timbre Models: 8 perceptual descriptors
  (hardness, depth, brightness, roughness, warmth, sharpness, booming,
   reverberation).
- Producer / synth-design vocabulary: sub bass, supersaw, ethereal pad,
  bell-like, pluck, etc.
"""

from __future__ import annotations

# Group -> list of (label, clap_query) tuples.
# Group names are what the user sees as section headers in the UI dropdown.
# We curated this list to drop jargon (nasal, watery, woody, focused, muted)
# and redundancies with the physical features (loud -> RMS, bright -> centroid).
VOCABULARY: dict[str, list[tuple[str, str]]] = {
    "vibe": [
        ("bright",      "a bright clear sound with high frequencies"),
        ("dark",        "a dark sound"),
        ("warm",        "a warm sound"),
        ("hard",        "a hard solid sound"),
        ("soft",        "a soft gentle sound"),
        ("deep",        "a deep sound from far below"),
        ("sharp",       "a sharp piercing sound"),
        ("booming",     "a booming sound with low frequencies"),
        ("rough",       "a rough harsh sound"),
        ("metallic",    "a brassy metallic sound"),
        ("hollow",      "a hollow empty sound"),
        ("sparkling",   "a sparkling brilliant sound"),
        ("airy",        "an airy breathy sound"),
        ("ringing",     "a ringing resonant sound"),
        ("percussive",  "a percussive sound with a sharp attack"),
        ("sustained",   "a sustained continuous sound"),
        ("reverberant", "a sound with lots of reverberation and space"),
    ],
    "type": [
        ("sub bass",       "a deep sub bass sound"),
        ("growl bass",     "a growling distorted bass sound"),
        ("reese bass",     "a reese bass sound"),
        ("wobble bass",    "a wobbling LFO bass sound"),
        ("supersaw lead",  "a supersaw lead with detuned sawtooth waves"),
        ("plucky lead",    "a plucky lead with a fast attack"),
        ("screaming lead", "a screaming aggressive lead"),
        ("glassy lead",    "a glassy bright lead"),
        ("ethereal pad",   "an ethereal otherworldly pad"),
        ("evolving pad",   "an evolving pad with slow modulation"),
        ("atmospheric pad","an atmospheric ambient pad"),
        ("drone",          "a drone sound"),
        ("bell-like",      "a sound like a bell"),
        ("mallet",         "a mallet sound"),
        ("pluck",          "a short plucked sound"),
        ("organ",          "an organ sound"),
        ("electric piano", "an electric piano sound"),
        ("riser",          "a riser sweep going upward"),
        ("impact",         "an impact hit"),
        ("noise fx",       "a noise effect"),
        ("arpeggio",       "an arpeggiated sequence"),
    ],
    "mood": [
        ("ambient",      "ambient music"),
        ("cinematic",    "cinematic music"),
        ("lush",         "a lush rich sound"),
        ("dreamy",       "a dreamy floating sound"),
        ("aggressive",   "an aggressive harsh sound"),
        ("gentle",       "a gentle delicate sound"),
        ("vintage",      "a vintage analog sound"),
        ("lo-fi",        "a lo-fi degraded sound"),
        ("otherworldly", "an otherworldly alien sound"),
        ("ethereal",     "an ethereal heavenly sound"),
    ],
}


def all_descriptors() -> list[tuple[str, str, str]]:
    """Flatten the vocabulary into (group, label, clap_query) tuples."""
    out: list[tuple[str, str, str]] = []
    for group, items in VOCABULARY.items():
        for label, query in items:
            out.append((group, label, query))
    return out


def labels() -> list[str]:
    return [label for _, label, _ in all_descriptors()]


def queries() -> list[str]:
    return [query for _, _, query in all_descriptors()]

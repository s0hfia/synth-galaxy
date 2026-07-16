# synth-galaxy

**A 3D-navigable latent space of Vital synth presets** — four independent bases (VAE of synthesis params, CLAP of perceptual audio, modulation structure, wavetable spectra), 60+ CLAP semantic descriptors for filtering, and drag-any-audio → find nearest presets → export a new `.vital` file using the extracted wavetable.

The pipeline works with **any Vital preset library you own** — this repo intentionally does *not* ship the presets themselves (most are covered by third-party licenses). Bring your own, run the scripts, get your galaxy.

---

## What it does

For a folder of `.vital` files, this pipeline:

1. **Renders each preset** to audio using headless [dawdreamer](https://github.com/DBraun/DawDreamer) hosting your installed `Vital.vst3`
2. **Generates mutations** — for each preset, N variants with small param perturbations so the latent space is dense enough to train a generative model
3. **Extracts audio features** (MFCC, spectral centroid, rolloff, flatness, stereo width, harmonic ratio) with librosa
4. **Trains a parameter VAE** with audio-consistency loss — encodes/decodes the 401-dimensional synthesis-param vector
5. **CLAP-embeds** each rendered preset via LAION-CLAP for perceptual semantic search
6. **UMAP-projects** to 3D across four bases:
   - **VAE latent** — organizes by *synth design similarity*
   - **CLAP audio** — organizes by *perceptual/how-it-sounds similarity*
   - **Modulation structure** — 4×4 source-family × destination-family fingerprint
   - **Wavetable spectra** — the actual oscillator harmonic content
7. **Serves an interactive Plotly viz** (Flask backend) where you:
   - Navigate between all 4 basis views
   - Color 2100 orbs by 60 CLAP descriptor scores (bright, dark, warm, ethereal, sub-heavy, bell-like, supersaw lead, aggressive, dreamy…)
   - **Click any orb** → sidebar with top-5 CLAP labels, 5 nearest sonic neighbors, top 10 z-scored standout params, active mod routings, 4×4 mod-fingerprint SVG, playback
   - **Drag any audio file onto the page** → CLAP-encoded, matched against your presets AND against a 66,904-wavetable corpus (WaveEdit Online + AKWF), extracts a wavetable from the audio, generates a fresh `.vital` file that lands in `~/Music/Vital/User/Presets/`

`src/synth_galaxy/preset_loader_full.py` loads `.vital` preset files into a **headless dawdreamer plugin instance**. Vital's `.vital` files are JSON but its VST3 state chunk uses JUCE's `VC2!`-magic-wrapped `MemoryBlock` base64 format. This module bridges them so you can programmatically apply any `.vital` preset without the Vital GUI. Details in [`PRESET_LOADER_NOTES.md`](PRESET_LOADER_NOTES.md).

## Prerequisites

- **macOS or Linux**
- **Python 3.12** (any Python 3.10+ probably works but 3.12 is tested)
- **[uv](https://github.com/astral-sh/uv)** for dependency management
- **Vital.vst3** installed at `/Library/Audio/Plug-Ins/VST3/Vital.vst3` (macOS default). If you install Vitalium instead — one-line swap in `src/synth_galaxy/config.py`
- **Your own Vital preset library** — most people already have `~/Music/Vital/` populated. Any folder of `.vital` files works; edit `src/synth_galaxy/config.py` to point elsewhere.

## Quick start

```bash
# 1. Install
uv sync

# 2. Point at your preset library (only if you don't have ~/Music/Vital/)
# Edit src/synth_galaxy/config.py -> PRESETS_ROOT

# 3. Full pipeline (2-4 hours end-to-end for a ~2000-preset library on M-series Mac)
uv run python scripts/render_presets.py           # ~1 h for 2000 presets
uv run python scripts/render_mutations.py         # ~3 h for 2000 × 20 mutations
uv run python scripts/extract_features.py         # ~5 min
uv run python scripts/train_vae.py                # ~15 min (M4 MPS)
uv run python scripts/clap_encode.py              # ~10 min
uv run python scripts/umap_clap_basis.py          # <1 min
uv run python scripts/umap_mod_basis.py           # <1 min
uv run python scripts/extract_wavetable_fingerprints.py
uv run python scripts/umap_wt_basis.py            # <1 min
uv run python scripts/precompute_details.py       # <1 min
uv run python scripts/galaxy_interactive_v2.py    # <1 min

# 4. Serve + open
uv run python scripts/search_server.py
# → open http://localhost:8000/
```

## Layout

```
src/synth_galaxy/
├── config.py                 # paths + rendering defaults
├── render.py                 # dawdreamer + Vital headless engine
├── sampler.py                # canonical synthesis-param subset (~400 of 2855)
├── mutation_sampler.py       # perturb-around-real-preset variant generator
├── preset_loader_full.py     # VC2! state wrapper for .vital loading (novel)
├── features.py               # librosa audio feature extraction
├── embed.py                  # UMAP wrappers
├── vae.py                    # param VAE model + training
├── clap_descriptors.py       # curated 60-descriptor vocabulary
└── dataset.py                # PyTorch datasets

scripts/                      # every stage of the pipeline as a runnable script
data/                         # generated (gitignored — bring your own preset library)
presets/                      # optional local staging (gitignored)
td/                           # TouchDesigner front-end (future)
```

## The audio-drop workflow

Once you have a galaxy running, drop any audio file onto the browser page:

1. **CLAP-encodes** the audio and pinpoints where in the CLAP basis your sound "lives"
2. **Finds the 5 nearest presets** in your library by cosine similarity
3. **Extracts a wavetable** from the audio (16 windows × 2048 samples, Vital's frame format)
4. **Matches against the 66,904-wavetable corpus** (WaveEdit Online + AKWF public-domain libraries)
5. **Generates a `.vital` preset file** using the extracted wavetable + FX/modulation defaults inherited from your nearest neighbor, saved to `~/Music/Vital/User/Presets/`

Producer workflow: love a stem in a Porter Robinson track? Drop it. Ten seconds later you have a real, editable, playable `.vital` file using a wavetable extracted from that exact audio, sitting alongside the nearest matches from your own library. Open Vital, load it, tweak it.

## Why the presets aren't in this repo

Most `.vital` preset packs (Florixel, SNFK, Cool WAV, Cosmos, Cyberwave, S1gns Of L1fe, Venus Theory, etc.) are covered by their designers' licenses — free-for-personal-use but not redistributable. Publishing them here would violate those licenses. Instead: bring your own preset library — the code works with whatever collection you own.

## External corpus attribution

The 66,904-wavetable reference corpus used for the audio-drop feature comes from:
- **[WaveEdit Online](https://github.com/smpldsnds/wavedit-online)** — 45,120 wavetables (CC0 public domain)
- **[Adventure Kid Wave Forms (AKWF-FREE)](https://github.com/KristofferKarlAxelEkstrand/AKWF-FREE)** — 21,784 single-cycle waveforms (CC0)

Both are downloaded on demand — see `scripts/` for the fetcher.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for what's built and what's next (bidirectional decoder → drop-marker synthesis, semantic drag, multi-parent blend, TouchDesigner sculpt UI).

## License

Code: MIT.

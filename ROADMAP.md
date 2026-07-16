# synth-galaxy — sculpt-mode roadmap

Where we are: a navigable galaxy of 2100 real Vital presets (plus 44k mutation
variants) in 3D, with VAE-latent and CLAP-perceptual basis swap, color-by feature
or CLAP descriptor, click-to-preview audio, and a detail panel showing top CLAP
descriptors / sonic neighbors / standout params / modulation graph.

Where we're going: a **visual synth shaper** where the user navigates the
cloud and generates *new* sounds — closer to a region, farther from it, married
between two presets, or steered by natural-language descriptors.

This document is the sequenced build plan to get there.

---

## Phase 0 — Verify the VAE decoder produces musical patches

Status: VAE trained on the 44k mutation dataset. Decoder exists but has not
been validated end-to-end (latent point → real Vital patch → audible audio).

**Tasks:**
- Sample 20 latent points uniformly inside the cloud, decode → params, write a
  `.vital` file, render via dawdreamer, listen.
- Pass criteria: ≥ 80% are audible (RMS > 0.005), ≥ 50% sound musically coherent
  (subjective; we listen).
- If quality is bad: retrain VAE with audio-consistency loss (the audio encoder
  branch we sketched earlier), or boost training data (more variants per preset,
  higher beta annealing).

Effort: half a day to test, up to several days if retraining is needed.

---

## Phase 1 — Backend: Flask `/decode` and `/save` endpoints

Extend `scripts/search_server.py` (already has `/search` for CLAP text→scores)
with two new endpoints.

```
POST /decode
  body: { latent: [12 floats] }  OR  { coords3d: [x,y,z] }
  returns: { wav_url: "/cache/abc123.wav",
             params: { ... 401 candidate values ... },
             features: { rms, centroid, ... },
             top_clap: [ {label, score}, ... ] }
  flow:
    - if coords3d: invert the 3D→12D map (we trained a small MLP for this earlier)
    - decode params via VAE decoder
    - set params on dawdreamer Vital instance
    - render 2s audio at C4, save to cache/<hash>.wav
    - extract librosa features + CLAP-encode the audio → top descriptors
    - return URL + metadata

POST /save
  body: { latent: [12 floats], filename: "my_patch.vital" }
  returns: { path: "/Users/.../my_patch.vital" }
  flow:
    - same decode, but write the full .vital JSON file via the inverse of
      preset_loader_full.py (params → JSON; wavetables inherited from a "base"
      preset chosen by nearest neighbor in the latent)

POST /steer
  body: { latent: [12 floats], directions: [ {term: "bright", weight: +0.3}, ... ] }
  returns: { new_latent: [12 floats] }
  flow:
    - For each term, compute its CLAP direction in latent space:
      (mean of audio_embs of top-K presets scoring high on that term)
      minus (mean of all audio_embs), then mapped through audio-to-latent regressor.
    - Sum weighted directions, add to input latent, return.
```

The audio cache (`data/sculpt_cache/`) uses a hash of the latent vector as the
filename so repeated decodes are instant. Garbage-collect entries older than a day.

Effort: ~1 day.

---

## Phase 2 — UI: Drop-marker (single-point synthesis)

Click in empty 3D space → spawn a marker → backend decodes → autoplay.

**Implementation:**
- Plotly emits `plotly_click` only when a marker is clicked, not on empty space.
  Workaround: add a transparent full-window event listener that captures clicks
  outside of any trace point and converts the screen position to scene 3D coords
  via `Plotly.relayout` math (look up `scene.camera`, `scene.aspectratio`,
  apply ray-cast through the plot's projection matrix).
- Simpler MVP: add a "**place marker**" button in the side panel that grabs the
  current camera target as the marker position.
- On marker placement:
  - POST `/decode` with the 3D coords.
  - Plot the marker as a glowing third trace (different color/symbol so it stands out).
  - Audio player auto-loads the returned wav_url and plays.
  - Detail panel populates with the decoded patch's CLAP descriptors and
    standout params (same renderer as the existing click panel, with a
    `"(generated)"` badge).
- Side panel adds **"Save .vital"** button. On click: POST `/save`, prompt
  filename, show "Saved to ~/projects/synth-galaxy/data/sculpt_exports/X.vital".

Effort: ~1 day.

---

## Phase 3 — UI: Two-point morph (marry two synths)

Pick two orbs → slider for interpolation 0→100% → continuously decode and play
intermediate points.

**Implementation:**
- Add **"morph from / to"** affordance: shift-click an orb to mark it as A,
  shift-click another to mark it as B. Visualize a faint line between them.
- Side panel section "MORPH" with:
  - Two preset names (A and B), each clearable.
  - Slider 0–100. As it moves, fetch `/decode` with the interpolated latent
    `(1-t) * latent_A + t * latent_B`. Debounce to avoid spamming the server
    (or use OSC-style streaming endpoint).
  - "Save current morph" button → POST `/save` with the current latent.
- Show audio waveform thumbnail next to A, B, and the morph result so you can
  see the timbre evolution.

Effort: ~1 day.

---

## Phase 4 — UI: Semantic drag (steering by CLAP terms)

Type "+bright" or "-warm" → backend computes the CLAP direction in latent space
→ pushes the marker → re-decodes → new patch that's measurably more "bright" or
less "warm".

**Implementation:**
- Side panel section "STEER" with:
  - Text input: "describe a direction (e.g. +bright, -warm, +ethereal)".
  - On Enter, parse terms (`/[+-]?\w+/`), POST `/steer` with the current marker
    latent and the parsed directions.
  - Backend returns new latent → POST `/decode` → marker moves, audio plays.
  - Show "moved by" delta: `bright +0.43, warm -0.21` so user knows what shifted.
- Stack multiple directions visually:
  - `+ethereal` `-aggressive` `+reverberant` (each chip removable).
- "Undo last steer" button keeps a stack of past latents.
- "Reset to nearest preset" button snaps to the closest real preset.

Effort: ~1 day.

---

## Phase 5 — UI: Multi-parent blend (3–5 source weighted average)

Select 3–5 orbs → sliders forced to sum to 1 → decoder renders the weighted blend.

**Implementation:**
- Side panel section "BLEND" with up to 5 slots. Each slot is a preset name
  (cmd-click an orb to add it to the next empty slot) and a slider 0–100.
- Renormalize sliders so they sum to 100.
- On any change: POST `/decode` with `sum(w_i * latent_i)` → audio plays.
- Show the dominant CLAP descriptor of the blend in real time so user
  understands the hybrid.

Effort: ~1 day.

---

## Phase 6 — UI: Live brush (sculpt-mode "flying through sound")

Drag the camera around with a marker attached → continuous decode → continuous
audio updates. Feels like flying.

**Implementation:**
- "Brush mode" toggle. When ON: marker locks to camera focal point. As the
  camera orbits/pans/zooms, sample a new latent every ~250ms, send to backend
  via WebSocket (cheaper than POST), receive audio chunk, schedule playback
  with crossfade.
- Or simpler MVP: poll `/decode` every 500ms on camera-stop, autoplay each.
- This is the most "synesthete" experience — visual movement directly drives
  sound.

Effort: ~2 days (WebSocket streaming is the slow part; polling MVP is ~1 day).

---

## Phase 7 — Export .vital with full fidelity

Right now decoder outputs 401 normalized param values. To write a real
loadable `.vital` file we also need:
- Wavetables: inherit from the *nearest neighbor* base preset (in CLAP space).
  This way the generated patch sounds plausible because it uses musical
  wavetables, not Vital defaults.
- Modulation routings: same — inherit from nearest neighbor.
- LFO shapes: same.

This means `/save` becomes: decode params → find nearest base preset by CLAP
similarity → copy wavetables/LFOs/mods from that preset's JSON → override
the 401 flat params with decoded values → write JSON.

Effort: half a day.

---

## Phase 8 — VAE retrain with audio-consistency loss

The current VAE was trained on params only. Adding the audio encoder branch
with an agreement loss lets us:
- Drop arbitrary audio clips into the galaxy (the audio encoder maps them to
  the same latent space).
- "Sounds like this Skrillex stab" → search the galaxy.

**Implementation:**
- Add `AudioEncoder` (CNN over mel-spec → 12D latent).
- Joint training: ParamVAE loss + audio-reconstruction loss + agreement loss
  `|| E_param(p) - E_audio(audio(p)) ||²`.
- 44k paired (params, melspec) examples already exist.

Effort: ~3 days (architecture + training + tuning).

---

## Open questions / future bets

- **TouchDesigner front-end:** shaders, particle effects, OSC bridge to the
  Flask backend. The roadmap above is a web app; TD is the same architecture
  but with the prettiest possible 3D rendering. 1–2 weeks of polish layer.
- **Multi-note sampling:** render every preset at C2/C3/C4/C5/C6 → trajectories
  in latent space → galaxy points become *lines* showing key-tracking behavior.
- **Scrape GitHub for more .vital files:** maybe 1–3k more presets findable in
  public repos. Background job, anytime.
- **Foundation model scaling:** multi-synth (Vital + Surge + Dexed share a
  latent), text-conditioned (CLIP-style on patch names + descriptions), audio
  reference at scale (encoder trained on million-clip corpora).

---

## Suggested build order

1. Phase 0 (verify decoder) — gates everything else
2. Phase 1 (Flask backend) — unblocks all UI work
3. Phase 2 (drop-marker) — the first user-facing magic moment
4. Phase 3 (two-point morph) — the "marry synths" feature
5. Phase 4 (semantic drag) — the "intelligence layer" payoff
6. Phase 7 (full-fidelity export) — makes everything actually shippable to Vital
7. Phase 5 (multi-blend) and Phase 6 (live brush) — power-user features
8. Phase 8 (audio-consistency VAE) — research / scaling

End-to-end MVP (phases 0–4 + 7): roughly **1 week of focused work**.

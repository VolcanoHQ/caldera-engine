# Caldera Engine Production Knowledge & Media Generation Roadmap

*Status: authored 2026-07-04; updated same day after Chains C and D shipped. Layers 1-3 are
live end-to-end (first full Tier 3 master produced for Peter Rabbit); Layer 4's audio chains
(C: MusicGen music + AudioLDM layered SFX; D: timeline assembly/mixing) are live; image/video
chains (E/F) remain design-only. The per-AI accounting — every pass, its system prompt,
provider policy, and validation — lives in `Caldera Engine AI Roster & System Prompts.md`.*

The goal: turn a raw manuscript into a **production knowledge base** rich enough to drive not just multi-voice audiobooks, but Graphic-Audio-style full productions (music, SFX, ambience) and eventually images/video — with every generation stage consuming the same structured scene/character records rather than re-analyzing text.

---

## 1. The Layer Model

```
Layer 1  TEXT STRUCTURE      (live)   tier_1_parser loops 1-4: parts/chapters/scenes/lines
Layer 2  SEMANTIC ENRICHMENT (live)   LLM passes: speaker attribution, emotion, utterance
                                      type, vocalizations, grounded SFX cues, alias merge,
                                      clean-check
Layer 3  PRODUCTION DIRECTION (built) scene_director: music direction, scene environment,
                                      delivery notes, generation prompts, MemPalace sync
Layer 4  MEDIA GENERATION    (design) prompts -> AI models -> assets -> assembled media
```

Rule learned the hard way (three abandoned parallel pipelines in this repo's history):
**every layer extends the live path behind a flag; no new disconnected siblings.**

---

## 2. Storage: MemPalace as the single production knowledge base

MemPalace (SQLite `data/mempalace/palace_relational.db` + Chroma vectors) already has the
right tables. Mapping of every extracted detail to its storage home:

| Detail | Where it lives | Written by |
|---|---|---|
| Scene identity, chapter, title | `wings` row | scene_director sync |
| Scene environment (location/time/weather/confines/noise) | `wings.metadata_json.environment` | scene_director (Layer 3 LLM pass) |
| Music direction (mood, style, stingers) | `wings.metadata_json.music` | scene_director |
| Grounded SFX cues (timeline-anchored) | `wings.metadata_json.sfx_events` | scene_director (from Layer 2 cues) |
| Generation prompts (music/ambience/sfx/image) | `wings.metadata_json.generation_prompts` + `tier3/generation_prompts.json` | scene_director prompt builder |
| Character canonical identity | `drawers.character_name` | register_character |
| Character aliases (per book) | `confirmed_merges` | scene_director sync (from Layer 2 alias merge) |
| Character voice (reference wav / builtin speaker / timbre embedding) | `drawers.voice_ref_path`, `drawers.base_embedding` | voice pipeline |
| Character per-emotion voice variants | `emotional_references` | voice pipeline (future) |
| Character visual profile (appearance notes for image gen) | `drawers.modulation_config_json.visual_profile` (additive key) | scene_director (future pass) |
| Per-line: speaker, text, emotion, confidence, performance, utterance_type | `rooms` + `rooms.metadata_json` | pipeline sync |
| Per-line rendered audio path | `rooms.audio_output_path` | voice_synthesizer |

File artifacts remain the debug/interchange layer (one dir per book):
`data/corpus/pipeline/{book}/tier1/*` (Layers 1-2) and `.../tier3/*` (Layer 3):
`production_script.json` (machine), `production_script.txt` (human-readable, directly
comparable to the `HumanProcessed/Tier 3` gold format), `generation_prompts.json`.

---

## 3. Character tracking: identity spine

1. **Within a book**: Layer 2's book-level roster + alias-merge pass produces canonical
   identities; `confirmed_merges` persists them keyed by book filename, so re-runs and
   downstream stages agree on who is who. `is_confirmed=0` rows record *rejected* merges,
   preventing the LLM re-proposing them (human veto is durable).
2. **Voice binding**: one `drawers` row per canonical character. Voice = cloned reference
   wav when available, else deterministic XTTS builtin speaker (stable hash, gender-pool
   heuristic). Timbre embeddings enable later voice-consistency checks.
3. **Visual binding (future)**: same drawer, additive `visual_profile` key — physical
   description extracted once per book, reused in every image/video prompt so the
   character looks consistent across scenes. Same pattern as voice: extract once, bind
   to canonical identity, reference everywhere.
4. **Cross-book identity** (future): `confirmed_merges` is book-scoped by design; a
   series-level identity table is a deliberate later step, not accidental scope.

## 4. Execution chains (mini-pipelines)

Every chain is: **structured input → (LLM or model) → schema-validated output →
programmatic grounding check → storage**. Validation is enforcement; prompts are only
suggestions (lesson: an 8B model copied the prompt's own few-shot example until a
verbatim-grounding check made that impossible).

```
CHAIN A (text->script, live):
  raw.txt -> loops 1-4 -> attribution/emotion/SFX (Groq bulk, Gemini quality-reserve,
  Ollama offline) -> alias merge -> ManuscriptManifest

CHAIN B (script->direction, built):
  manifest + scene text -> scene_director LLM pass (music/environment/delivery) ->
  grounding validation -> production_script + generation_prompts + MemPalace sync

CHAIN C (direction->audio assets, next):
  music_prompt   -> MusicGen (open weights, local)   -> scene music bed
  ambience_prompt-> AudioGen / CC libraries (UCS naming) -> ambience loop
  sfx_prompts    -> AudioGen / Freesound CC0          -> per-event SFX
  (placeholder-first: pipeline runs with silence/existing 2 assets before real ones)

CHAIN D (assembly, mixer exists):
  line WAVs (XTTS, live) + music bed + ambience + timeline-anchored SFX
  -> AudioMixer (sidechain ducking under voice, overlay, ACX mastering) -> master

CHAIN E (direction->images, future):
  image_prompt (setting + characters-present w/ visual profiles + mood + style)
  -> SDXL/Flux local -> scene illustration; character sheets first for consistency

CHAIN F (images->video, future-future):
  scene stills + motion prompts -> SVD/LTX-class local video models -> animated scenes
```

Timeline anchoring for Chain D: every SFX/stinger references a `line_id`; line WAV
durations are known post-synthesis, so cumulative offsets convert "after line N" into
timestamps mechanically. No LLM in the assembly path.

## 4b. Voice Marketplace lane (implemented 2026-07-05)

Seller → buyer → production flow, all live in `src/voice_marketplace.py`:
`onboard` (samples → denoise via `voice_synthesizer.denoise_audio_file` → validated ≥6s
mono cloning reference → Qdrant listing with seller/consent/price provenance) →
`purchase_voice` (append-only license ledger: what/who/purpose) → `cast_character`
(description-based search → license → bind to the character's MemPalace drawer, so every
synthesis uses the purchased voice; zero-shot XTTS conditioning makes the "training"
instant, per-voice fine-tuning is the future premium tier). The visual analog is
deliberate: a purchased "character look" is a LoRA/reference-sheet the way a purchased
voice is a reference set — Chain E's identity-lock (IP-Adapter conditioning on a sheet)
is the first step of that symmetry.

## 4c. Cross-book identity (design)

Per-book identity is settled by `confirmed_merges` (book-scoped, human-veto durable).
Series-level identity is a SEPARATE table by design: `series_identities(canonical_name,
book_filename, book_canonical_name, confirmed)` built OVER per-book merges, mapping e.g.
"Holmes" across twelve stories to one series entity (one voice drawer, one visual
profile). Two rules: (1) NEVER auto-merge across books — same-name characters in
different books are only candidates, and a human (or an explicit series manifest)
confirms; (2) the series entity owns casting (voice + look), the per-book entity owns
attribution. This keeps a false cross-book merge from poisoning multiple books at once.

## 4d. Chain F: video (design sketch)

Chain F consumes only existing per-scene assets — no new analysis AIs:
scene still (or identity-locked still) as the keyframe + the Music Director's mood/events
as the motion-energy curve + line-anchored timeline for cut points. Local model targets:
SVD-class image-to-video for 2-4s scene loops, LTX-class when VRAM allows. Assembly is
Chain D's pattern transposed: deterministic timeline, ffmpeg concat, audio master muxed
underneath. The production script literally becomes an animatic edit list. Prerequisite:
Chain E identity-lock quality must be accepted first — animating inconsistent characters
compounds the inconsistency.

## 4e. Tier 1 lane (single-narrator audiobook) — measured 2026-07-13

Tier 1 is the zero-LLM product: loops 1-4 structure + clean narration text + one
narrator voice. Text fidelity vs `HumanProcessed/Tier 1` golds: Peter Rabbit 99.5%,
Case of Identity 100.0%, Alice 99.8% coverage, all ≤0.31% contamination. The
ClutterScrubber now chops Gutenberg back matter (the license was previously read
aloud in 5 corpus books), handles `START OF THIS` markers, strips transcriber
credits, and a deterministic heading guard keeps the fuzzy detectors from eating
story text (they had swallowed Wuthering Heights' opening and Frankenstein's
Letters 1-2). Assembly: `production_mixer --tier1` — every line synthesized as the
Narrator drawer, concatenated with per-line padding + 1.5s chapter gaps, ACX
mastering. Line emotion is kept as subtle pitch/speed cadence (one voice, not a
cast) — the "Tier 1.5" inflection idea in its mildest form.

## 5. Testing strategy

- Each chain has a gold or a ground-check: Chain A vs `HumanProcessed Tier 1/2` (100%
  speaker accuracy on Peter Rabbit); Chain B's `production_script.txt` vs `HumanProcessed
  Tier 3` gold; Chain C/E judged by human review initially (add golds as they stabilize).
- Peter Rabbit is the canonical dev book (small, all three gold tiers). Sherlock corpus
  is the hard-attribution benchmark. Alice is the novel-scale benchmark.
- The zero-LLM Tier 1 stress test remains the regression gate for every change.

## 6. Current status & near-term order

1. ~~Layers 1-2~~ live (Tier 1 + enrichment; 100% attribution on Peter Rabbit gold; validated
   on the 12-story Sherlock corpus and Alice).
2. ~~Layer 3~~ live as a 4-role production crew (Spotter → Music Director / Sound Designer /
   Dialogue Director), each schema-validated and text-grounded.
3. ~~Chain D~~ live: line-anchored timeline, layered SFX composites, sidechain-ducked music,
   ACX mastering. Human-in-the-loop editing via `line_overrides.json` + `[pause:X]` markup +
   pinned `xtts_speaker` casting in MemPalace drawers.
4. ~~Chain C~~ live: MusicGen (beds/stingers, loudness-normalized) + AudioLDM v1 (SFX layers,
   composited by timing/level; per-scene generated ambience). All clips prompt-cached.
5. Wire delivery notes into synthesis (today only emotion → pitch/speed mods reach XTTS).
6. Remove SFX duty from the attribution AI (Sound Designer owns sound now) — see AI Roster.
7. Character visual profiles + Chain E character sheets (Peter Rabbit illustration test).
8. Revisit scene sub-segmentation (novel-scale chapters need finer scenes for direction).

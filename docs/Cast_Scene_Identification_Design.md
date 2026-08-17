# Design: Decouple Cast/Scene Identification from Production Tier

*Status: proposed, not yet implemented. Written 2026-08-16 in response to a
product question: "shouldn't the cast be identified no matter the tier, and
scenes too — letting a user run a single narrator whose pitch/tone is
adjusted per-character, with those adjustments reused automatically
wherever that character appears?"*

---

## 1. The problem, as it exists in the code today

Tier is currently one dial that conflates three independent decisions:

| What tier actually controls today | Where |
|---|---|
| Whether characters are identified at all | `hierarchical_parser.py:135-141` — `if self.production_tier == 1: global_characters = ["Narrator"]` (skips coreference/global character extraction entirely) |
| Whether dialogue lines are attributed to a speaker | `hierarchical_parser.py:480-486` — `if production_tier == 1: assigned_character = "Narrator"` (every line, unconditionally) |
| Whether LLM speaker-attribution enrichment runs | `render_job.py:200` — `enable_llm_enrichment=(tier >= 2)` |
| Whether the final mix uses one voice or many | `production_mixer.py:490,519-521` — `single_narrator` flag forcibly overwrites every line's `character` field back to `"Narrator"` right before synthesis, discarding any attribution that *did* exist |
| Whether Scene Director / sound design / dramatization run | Tier 3 only (`production_mixer.mix_production`, separate code path) |

Net effect: **Tier 1 books never get a cast, never get scenes attributed to
characters, and never get per-character voice differentiation** — not
because any of those things are expensive or Tier-3-only in nature, but
because the *code path itself* is short-circuited at Tier 1.

The Cast Manager UI (`index.html` `openCastManagerModal()`, ~line 3797)
already has fully-built per-character `speed`/`pitch` number inputs backed by
`/api/cast` and `MemPalace.register_character()`'s
`modulation_config` (`speed`, `pitch`, `volume`, `energy_bias`,
`prosody_stabilization`) — but it reads its character list from
`currentHierarchy.metadata.global_characters`, which is forced to `[]` /
`["Narrator"]` at Tier 1. The UI to do exactly what's being asked for
already exists; it's just unreachable at Tier 1 because the data it needs
was never computed.

## 2. Why the current model is wrong

Two genuinely different questions are being answered by one knob:

1. **"Do we know who is speaking, and where scenes/chapters break?"** —
   this is manuscript *analysis*. It doesn't cost extra narrator/voice-actor
   time and shouldn't depend on production budget. Every tier benefits from
   knowing "Arthur says this line, Emily says that one" even if they're all
   ultimately voiced by the same actor.
2. **"How many distinct voices/production values do we spend on the final
   render?"** — this is a *production* decision (cost, actor availability,
   desired listening experience). This is legitimately tier-shaped.

Collapsing (1) into (2) means a Tier 1 (single-narrator, cheapest) user gets
*zero* cast awareness, even though `MemPalace` already supports registering
many characters against **the same underlying voice reference wav** with
different `speed`/`pitch`/`energy_bias` — i.e., the infrastructure to make
one voice actor sound like a different character per speaker already
exists and is unused at Tier 1 today.

## 3. Proposed model

Split the single `tier` knob into two independent axes. Keep the name
"Tier 1/2/3" for the production-complexity axis (least disruptive to
existing UI/vocabulary/DoD ledger), and make cast/scene identification
always-on underneath it.

### 3.1 Always-on: manuscript analysis (no tier gate)
- Global character extraction / coreference resolution (currently gated at
  `hierarchical_parser.py:135`) runs for every tier.
- Per-line dialogue attribution (currently gated at
  `hierarchical_parser.py:480` and the LLM enrichment flag in
  `render_job.py:200`) runs for every tier. A Tier 1 book still ends up
  with `line.character == "Arthur"` where appropriate instead of always
  `"Narrator"`.
- Scene/chapter structure detection is already tier-independent
  (`nlp_analyzer.py`'s scene segmentation isn't gated the same way) —
  confirm and keep it that way.
- Cast Manager UI becomes available regardless of active tier/mode
  (currently hidden entirely in Express mode via `castManagementSection`
  display toggling in `switchGlobalTab`/`switchUIMode`).

### 3.2 New, separate knob: voice differentiation
Introduce an explicit `voice_mode` independent of tier:
- **`single_narrator_modulated`** (new default for Tier 1): every
  identified character gets its own `MemPalace` drawer, but all drawers
  point at the *same* `voice_ref_path` (the user's chosen single
  narrator). Each drawer's `speed`/`pitch`/`energy_bias` can be
  independently tuned via the existing Cast Manager sliders — so "Arthur"
  might be `pitch: -2.0` and "Emily" might be `pitch: +3.0`, both spoken by
  the same underlying voice actor/model, without ever cloning/hiring a
  second voice.
- **`multi_voice`** (today's Tier 2/3 behavior): each character gets its
  own distinct `voice_ref_path` (cloned reference, Neural voice, etc.).

`production_mixer.mix_voice_track`'s `single_narrator` boolean becomes this
`voice_mode` enum instead. Critically: **stop overwriting
`line["character"]` back to `"Narrator"`** (`production_mixer.py:519-521`)
— keep the real attributed character for drawer lookup and for anything
downstream that wants to know who's speaking (read-along transcripts, cast
lists, future analytics), and only use the shared reference wav as the
*voice*, not as the *identity*.

### 3.3 Tier becomes purely "production complexity"
- Tier 1: single-narrator-modulated voice, no Scene Director/sound
  design/dramatization.
- Tier 2: multi-voice cast, still no Scene Director/dramatization.
- Tier 3: multi-voice cast + full Scene Director/sound design/dramatization
  artifacts.
- A user could in principle also choose "Tier 3 dramatization, but with
  single-narrator-modulated voice" — that combination isn't required for
  v1 but the split makes it possible later without more rework.

## 4. Concrete changes required (implementation sketch, not yet done)

1. **`hierarchical_parser.py`**: remove the `production_tier == 1` bypass
   at line ~135 (global character extraction) and ~480 (per-line
   attribution forcing `"Narrator"`). Character extraction/attribution
   always runs; `production_tier` no longer read by these code paths at
   all.
2. **`render_job.py`**: change `enable_llm_enrichment=(tier >= 2)` to
   always `True` (confirmed decision — full LLM pass for every tier, see
   §5).
3. **`production_mixer.mix_voice_track`**: replace the `single_narrator`
   bool parameter with a `voice_mode: Literal["single_narrator_modulated",
   "multi_voice"]`. Drop the `d["character"] = "Narrator"` overwrite.
   Drawer registration loop: for `single_narrator_modulated`, register
   every identified character's drawer against the single chosen
   `voice_ref_path`, defaulting modulation to neutral unless the user
   customized that character via Cast Manager.
4. **`console_api.tier_readiness`**: `tier1.ready`/`message` semantics
   change slightly — Tier 1 readiness should now also reflect "cast
   identified" rather than just "line artifacts exist," since cast
   identification is no longer Tier-2-exclusive.
5. **GUI (`index.html`)**:
   - Un-hide `castManagementSection` for Express mode / Tier 1 (currently
     forced `display: none` in `switchUIMode`'s `'express'` branch).
   - Cast Manager's `wavOptions` list / per-character voice picker needs a
     "use narrator voice, just modulated" default row so users aren't
     required to pick a distinct reference file to get a differentiated
     character.
6. **Backward compatibility**: existing rendered Tier 1 books / cached
   `loop4_lines_enriched.json` artifacts were generated under the old
   bypass (`attribution_method: "Tier 1 Default"` everywhere) — these
   caches will look stale under the new cache-invalidation logic in
   `gui_server.py handle_post_analyze` (it already tracks
   `analysis_pipeline`/`book_structure_version` for exactly this kind of
   change) and should reprocess automatically on next analyze, not require
   manual cache-busting.

## 5. Decisions (confirmed with user 2026-08-16)

1. **LLM cost at Tier 1**: **full LLM attribution pass for all tiers** —
   same quality/cost model Tier 2/3 already pays today. No cheaper
   heuristic-only carve-out for Tier 1.
2. **Naming**: **`voice_mode` is its own separate concept from `tier`** —
   not folded into tier vocabulary. Confirmed values:
   `single_narrator_modulated` / `multi_voice`.
3. **Default per-character modulation**: **all-neutral by default.** A
   newly-identified character's `MemPalace` drawer gets neutral
   `speed`/`pitch`/`volume`/`energy_bias` (identical to Narrator) until the
   user manually tunes them via Cast Manager. No automatic hash-based
   pitch offset.
4. **Phasing**: **two phases**, landed and reviewed separately:
   - **Phase A (this change)**: remove the Tier-1 bypasses so cast/scene
     identification always runs and is stored/visible, regardless of
     `voice_mode`/tier. No rendering/mixing behavior changes yet —
     `production_mixer.py` keeps overwriting to `"Narrator"` for now, so
     actual Tier-1 audio output is unchanged in this phase.
   - **Phase B (follow-up)**: wire `voice_mode` into
     `production_mixer.mix_voice_track`, drop the `"Narrator"` overwrite,
     un-hide Cast Manager for Express/Tier 1, add the "use narrator voice,
     just modulated" default row.

## 6. Non-goals for this change
- Not changing Tier 2/3's existing multi-voice behavior.
- Not touching Scene Director / sound design / dramatization gating —
  those stay Tier 3-exclusive.
- Not part of this: the Voice Marketplace integration (separate product,
  separate concern — see `docs/MVP_STATUS.md` §3 backlog item T2-5).

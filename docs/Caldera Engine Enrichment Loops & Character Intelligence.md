# Caldera Engine — Enrichment Loops & Character Intelligence

*Architecture reference for the layered enrichment model and the character-profile
product direction. Agreed map before implementation — not a spec for any single loop.*

Status: v1 (design). Grounded in the codebase as of this writing; every "exists" claim
points at a real artifact or module. Where this doc and the code disagree, the code wins
and this doc is corrected.

---

## Core principles (established, load-bearing)

1. **Understanding is uniform across tiers; tiers differ in delivery.** Every loop below
   produces *manuscript understanding* that is identical whether the book renders as a
   Tier 1 single narrator, a Tier 2 cast, or a Tier 3 production. Tiers diverge only in
   how that understanding is *delivered* (one voice vs. many vs. full production). A loop
   that made higher tiers "understand more" would violate this.

2. **Detection vs. delivery — the guardrail.** Text-derived *classification* (which scene
   boundary, which speaker, which emotion) is character-agnostic and never biased by
   identity or cross-scene state. Character identity and continuity affect only *delivery*
   (how a already-classified thing is performed). This is not stylistic — biasing
   classification with character/history state is precisely what once collapsed whole books
   onto one emotion (a detective novel read 80% "cheery"). Guarded by
   `tests/test_emotion_distribution.py` and `tests/test_character_continuity.py`.

3. **Slicing is authoritative and comes first.** Scene slicing (deterministic + the G4
   Director's Scene Segmenter) is finalized before attribution/emotion, and the canonical
   structure is built from the same slice the line payloads use. Attribution and emotion
   consume those scenes and never re-slice. (Fixed this session; `structure_to_manifest`
   now fails loud on any payload scene absent from the structure.)

4. **Deterministic before LLM.** Each loop separates its deterministic substrate (indexes,
   participants, gaps, arcs) from its LLM-authored analysis (mood, purpose, relationships).
   The deterministic parts run at zero cost for all tiers — preserving Tier 1's offline
   guarantee — and are what the delivery layers and the future product consume.

---

## The 8-loop model

Conceptual architecture (the mental model). Physical artifact filenames are **not**
renumbered to match — see the numbering note below.

| Loop | Purpose | Status | Current home / new work |
|---|---|---|---|
| **L1 · Metadata extraction** | Front matter, illustrations, book-level bible | ✅ exists | `loop1_illustrations.json`, `book_bible.json`, `profile.json` |
| **L2 · Parts** | Part boundaries | ✅ exists | `loop1_parts.json` |
| **L3 · Chapters** | Chapter boundaries | ✅ exists | `loop2_chapters.json` |
| **L4 · Scenes by chapter** | Scene slicing (deterministic + G4) | ✅ exists (hardened) | `loop3_scenes.json`; authoritative, drop-guarded |
| **L5 · Character by scene** | Who is present/speaking per scene; appearance index | 🟡 started | `build_appearance_index` in `character_continuity.py`; attribution in `loop4_lines_enriched.json` |
| **L6 · Scene analysis** | Per-scene mood, tension, purpose, emotional arc | ⬜ new | boundary-v2 gives confidence/reasons only |
| **L7 · Character analysis** | Consolidated portable character profile | 🟡 partial, scattered | `character_profiles.json` (visual+expression), continuity + per-scene emotion in `performance_script.json` |
| **L8 · Storyboard / timeline** | Scene-ordered production timeline / storyboard | 🟡 partial | `mix_timeline` (audio), `character_sheets`/stills (image), `production_script.json` |

The performance intelligence built to date (attribution reduction → text-only emotion →
expression profile → character continuity + appearance index) is the connective tissue
between L4/L5 and delivery, and is emitted in `performance_script.json`.

**Numbering note.** The codebase already uses `loop1–4` as physical artifact names, but with
different boundaries than this model (current `loop4` = *lines*, this model's L4 = *scenes*).
Renumbering the files would churn every module that reads those paths. Keep this model as the
mental architecture; give the *new* loops (L5–L8) their own descriptively-named artifacts
rather than renumber the existing ones.

---

## L7 — the consolidated Character Profile (keystone)

Character data exists today but is scattered across three artifacts (visual +
`emotion_expression_profile` in `character_profiles.json`; appearance index, continuity, and
per-scene emotion in `performance_script.json`). L7 consolidates them into **one versioned,
portable per-character entity** — the thing every downstream product hangs off, and the unit
a shared database or marketplace could carry.

Proposed portable schema (deterministic assembly + existing LLM outputs; versioned):

```json
{
  "profile_version": "character_profile_v1",
  "identity":   { "name": "Sherlock Holmes", "aliases": ["Holmes"], "role": "primary" },
  "provenance": { "book_id": "...", "source_work": "...", "source_hash": "...", "designed_by": "..." },
  "visual":     { "visual_description": "...", "evidence": ["..."], "inferred": true },
  "expression": { "angry": "restrained", "cheery": "dry" },
  "arc":        { "appearances": [ { "scene_id": "...", "position": 3, "emotion": "Tense" } ],
                  "dominant_emotion": "Tense", "emotional_range": ["Tense","Flat","Angry"] },
  "voice_affinity": { "marketplace_query": "Sherlock Holmes: gaunt, hawk-nosed detective ...",
                      "current_voice_ref": null },
  "consent":    { "shareable": false, "license": null }
}
```

- `visual`, `expression` already exist. `arc` is a trivial derivation from the appearance
  index we now emit. `voice_affinity.marketplace_query` is exactly what `_character_voice_query`
  already computes — the profile→voice-selection link is already live.
- `consent` mirrors the voice marketplace's consent-gated model (`voice_marketplace.onboard_voice`)
  and defaults to private; nothing is shared unless the uploader opts in.

---

## Product direction — the character database

If an uploader opts a book's characters public, the profiles feed a shared character/story
database — a potential standalone product (character progression, character search) that also
powers downstream selection:

- **Voice-actor selection** — `voice_affinity` → the existing voice marketplace query (already
  wired).
- **Actor / casting selection** — same profile, richer query.
- **AI character personality** for the interactive tier — `identity` + `expression` + `arc` as
  a persona seed.

The reusable pattern already exists: the voice marketplace's consent-gated onboarding + license
ledger (DoD F-10). A character marketplace parallels it rather than reinventing it.

### Open decisions (gate the public database — decide before building it)

1. **IP / copyright.** Clean for public-domain works (Sherlock, Peter Rabbit); fraught for
   characters extracted from a user's *copyrighted* upload. `consent.shareable` must encode
   "make public only what you have rights to," not a bare checkbox. *Recommendation:* default
   private; public opt-in gated on an explicit rights attestation; treat public-domain and
   user-original works as the initial shareable set.
2. **Cross-work character identity.** Is "Holmes" in book A the same DB entity as book B?
   Merging/deduplicating across works is the hard core of a character-DB product.
   *Recommendation:* v1 keeps profiles **per-work** (no cross-work merge); a DB entity is
   `(source_work, character)`. Cross-work identity is a later, explicit feature.
3. **Schema & provenance.** A shareable profile must travel without book-specific state and be
   versioned. *Recommendation:* the `profile_version` + `provenance` block above; never embed
   raw manuscript text in a shareable profile (evidence snippets are short quotes only).

---

## Sequencing

1. **L7 consolidated Character Profile** (local artifact, no sharing) — keystone; mostly
   deterministic assembly of existing signals + the schema above. Unlocks character analysis
   and is the entity the product needs.
2. **L6 Scene analysis** — per-scene mood/tension/purpose/arc; independent; understanding-tier.
3. **L5 completion** — richer per-scene cast state (present vs. mentioned, entrances/exits) on
   top of the appearance index.
4. **Cast / relationship enrichment** — per-scene active relationships (the deferred O(n²) LLM
   dimension); depends on L5/L7.
5. **Character database + consent/marketplace** — only after the open decisions above are made.

Each loop keeps the guardrails: understanding uniform across tiers, classification never biased
by identity/continuity, delivery modifiers deterministic and clamped.

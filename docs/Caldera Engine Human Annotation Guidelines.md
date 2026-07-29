# Caldera Engine — Human Annotation Guidelines (v1)

*How human reviewers create gold-standard reference data for evaluating manuscript
understanding: structure, speaker attribution, attribution reduction, emotion, expression
profiles, and production intelligence.*

The goal is **consistency and repeatability** across books and annotators, and — critically —
gold that compares **field-for-field** against what the engine actually emits. Every dimension
below ends with the exact system artifact/field it is scored against. When the guidelines and
the code disagree, the code's vocabulary wins (see the [System Vocabulary Reference](#system-vocabulary-reference));
this document is reconciled to the implementation as of guideline v1.

> **Tier model note.** Manuscript *understanding* (all six dimensions here) is uniform across
> tiers — tiers diverge in delivery medium and presentation, not comprehension. So a single
> gold set for a book scores every tier's understanding; only Dimension 6 (production) is
> tier-specific. This mirrors the engine: the same `performance_script.json` feeds the Tier 1
> single narrator, the Tier 2 cast, and the Tier 3 full production.

---

## General principles

**Rule 1 — Annotate the text, not your assumptions.** Label what is present in the manuscript.
Do not use outside/franchise knowledge, assume future events, or infer hidden author intent.
- ✗ "Holmes is usually calm, so this is restrained."
- ✓ "This line reads fearful because of the words on the page."

**Rule 2 — Evidence first.** Every non-obvious annotation carries a verbatim supporting snippet.
```json
{ "expression": "restrained", "evidence": ["Holmes clenched his jaw but said nothing."] }
```

**Rule 3 — Character-agnostic emotion.** Emotion is assigned from **line text only** — never from
character identity, reputation, or prior lines. *"I'm afraid."* is `fearful` no matter who says
it. (This mirrors the engine, which deliberately forbids character bias in emotion *detection*;
character identity only shapes *delivery*, in Dimension 5.)

---

## Dimension 0 — Segmentation gold

Everything aligns to segmentation; the evaluation harness matches predictions to gold by
normalized text within a scene, so if segmentation disagrees, every downstream score misaligns.

```json
{ "line_id": "line_001", "scene_id": "scene_001", "segment_type": "dialogue", "text": "…" }
```

**Segment types (gold may be richer than the system).** Annotate the fine-grained type when
useful — `dialogue, narration, thought, letter, journal, epigraph, other` — but record the
**system mapping** alongside, because the engine only distinguishes two `segment_type` values
plus an `utterance_type`:

| Gold type | System `segment_type` | System `utterance_type` |
|---|---|---|
| dialogue (spoken aloud) | `dialogue` | `speech` |
| narration | `narrative` | `speech` |
| thought / letter / journal / epigraph | `narrative` (or `dialogue` if voiced) | `speech` |
| non-lexical vocalization (gasp, sob) | `dialogue` | `vocalization` |

**Scored against:** `performance_script.json → scenes[].lines[] {line_id, scene_id, segment_type, text}`.

---

## Dimension 1 — Structure gold

Capture part / chapter / scene boundaries and any scene titles. A scene boundary is allowed on:
time change, location change, POV change, explicit separator, or major narrative transition —
**and must carry a reason**:

```json
{ "scene_id": "scene_014", "title": "The garden", "scene_break_reason": "location_shift" }
```

**Scored against:** `book_structure.json` (parts/chapters/scenes, titles) and
`tier1/loop3_boundaries.json` (the boundary-v2 confidence + reason codes). Metrics: boundary
precision / recall / F1; scene-count match; verbatim text-fidelity (contamination %).

---

## Dimension 2 — Attribution gold

Per dialogue line, the speaking character and their aliases:

```json
{ "line_id": "line_042", "speaker": "Sherlock Holmes", "aliases": ["Holmes", "he"], "uncertain": false }
```

Speaker must be explicit or inferable from immediate context; set `"uncertain": true` rather than
guessing. Maintain a book-level **character roster** with the canonical name and all aliases —
the engine surfaces plausible-but-different names, so alias resolution is scored separately.

**Scored against:** `line.character` / `line.speaker_id`; roster vs `tier1/loopE_llm_alias_merges.json`.
Metrics: speaker accuracy, alias-resolution accuracy.

---

## Dimension 3 — Attribution reduction gold

For **every attribution tag** in the text, record the decision a spoken audiobook should make:

- **remove** — redundant; speaker already unambiguous. *"'I agree,' Holmes said."* → drop the tag.
- **keep** — required for clarity. *"'I agree,' he said."* with multiple male speakers present.
- **metadata** — an expressive verb carrying delivery info; convert rather than read aloud.
  *"Holmes whispered."* → mark the line `whisper`.

```json
{
  "line_id": "line_042",
  "classification": "convertible",     // required | redundant | convertible
  "action": "metadata",                // keep | remove | metadata
  "delivery": "whisper",               // only for metadata; from the delivery set below
  "reason": "expressive verb carries delivery information"
}
```

**Decision rule for `keep`:** a tag is *required* only if removing it would make the speaker
ambiguous **to a listener who cannot see the page**, or if it carries staging/emotion beyond the
speaker's name. Everything else with a resolved speaker is `remove`.

**Delivery vocabulary (for `metadata`):** `whisper, mutter, shout, scream, hiss, snap`.

**Scored against:** `performance_script.json → reductions[] {classification, action, delivery, reason}`
and per-line `attribution_reduction` / `attribution_delivery`. Metrics: remove-precision,
keep-precision, metadata/delivery-extraction accuracy.

---

## Dimension 4 — Emotion gold

**Allowed labels (exactly these — the system's vocabulary):**
`flat, cheery, sad, angry, violent, tense, fearful`.

Choose **`flat`** unless there is meaningful emotional evidence in the words. Emotion is
text-driven and character-independent (Rule 3).

```json
{ "line_id": "line_042", "emotion": "fearful", "evidence": ["I'm afraid to go in there."] }
```

**Distribution analysis (required, per book).** Because the characteristic failure is
*distributional* (an emotion detector collapsing a whole book onto one label), record the
aggregate too:

```json
{ "flat": 82, "tense": 8, "fearful": 4, "sad": 3, "angry": 2, "cheery": 1, "violent": 0 }
```

This catches emotion collapse, over-classification, and systemic bias that per-line spot checks
miss. Two independent reviewers label emotion; agreement is recorded and disagreements adjudicated.

**Scored against:** `line.emotion` and `performance_script.json → emotion_pass`. Metrics: per-line
accuracy, confusion matrix, **distribution similarity** (the guardrail dimension).

---

## Dimension 5 — Expression gold

Character-level, and a different question from emotion: **how does this character *express* an
emotion once they feel it** — not what they usually feel. Keyed by the **canonical emotion labels**
(not free-form nouns), with evidence, confidence, and an `inferred` flag:

```json
{
  "character": "Sherlock Holmes",
  "emotion_expression_profile": {
    "angry":  { "style": "restrained", "confidence": 0.85, "inferred": true,  "evidence": ["…clenched his jaw but said nothing."] },
    "cheery": { "style": "dry",        "confidence": 0.70, "inferred": true,  "evidence": ["…a thin, sardonic smile."] },
    "fearful":{ "style": "suppressed", "confidence": 0.60, "inferred": true,  "evidence": [] }
  }
}
```

**Emotion keys must be the six canonical labels** (`cheery, sad, angry, violent, tense, fearful`),
one entry each — *not* overlapping nouns. Do **not** write both `humor` and `joy`: the engine
folds both onto `cheery`, so they collide and one silently overwrites the other. (The canonicalizer
does alias common nouns — anger→angry, fear→fearful, joy→cheery — but relying on that invites
exactly this collision; use the canonical keys directly.)

**Expression style vocabulary (exactly the engine's — anything else is dropped):**

| Band | Styles |
|---|---|
| Reserved | `hidden`, `internalized`, `suppressed`, `restrained`, `subtle`, `dry`, `quiet` |
| Balanced | `controlled`, `measured` |
| Intensified | `open`, `expressive`, `overt`, `intense`, `explosive` |

Styles outside this set (e.g. "natural", "conversational") are **not** recognized and produce no
delivery change — if a character's expression of an emotion is simply unremarkable, omit that
emotion from the profile rather than inventing a style. Common synonyms are auto-aliased
(reserved→restrained, wry→dry, volcanic→explosive, calm→controlled, …), but prefer the canonical
term.

**Scored against:** `character_profiles.json → emotion_expression_profile` (designer output) and
per-line `expression.{expression_style, intensity}`. Metrics: human-vs-designer agreement,
evidence-support rate, confidence calibration.

---

## Dimension 6 — Production gold (advanced tiers only)

Music cues, SFX cues (grounded in text), dramatized inserts, delivery direction, and — for the
media tiers — cinematic direction. This is the only dimension that is tier-specific.

```json
{ "scene_id": "scene_014", "music_cue": "…", "sfx_cue": "…", "dramatic_direction": "…", "camera_direction": "…" }
```

**Scored against:** `tier3/{sound_design.json, production_script.json, generation_prompts.json}`.
Metrics: human rating, cue accuracy, placement accuracy. (Existing Tier 3–5 HumanProcessed golds
model this today.)

---

## Annotation metadata

Every gold file carries provenance so scores are reproducible and comparable across books:

```json
{ "book_id": "", "source_hash": "", "annotator": "", "guideline_version": "1.0", "date": "" }
```

`source_hash` is over the exact source text (so text-fidelity is reproducible); `guideline_version`
must match the rules the annotator used — subjective labels only compare across books when the
rules were identical.

---

## Recommended workflow

| Stage | Reviewer(s) | Output |
|---|---|---|
| 1 · Structure | Reviewer A | `structure_gold.json` |
| 2 · Attribution | Reviewer B | `attribution_gold.json` |
| 3 · Attribution reduction | Reviewer B | `reduction_gold.json` |
| 4 · Emotion | **Two** independent reviewers + adjudication | `emotion_gold.json` |
| 5 · Expression | Senior reviewer (major cast; evidence + confidence + inferred) | `expression_gold.json` |
| 6 · Production | Advanced-tier evaluators | `production_gold.json` |

Emotion (Stage 4) and expression (Stage 5) are subjective; single-annotator gold is not trusted
for them — record inter-annotator agreement.

**Templates & tooling.** Fill-in JSON templates for every dimension live in
[`annotation_templates/`](annotation_templates/) (one per gold file, with the `_meta` block and
illustrative entries). The expression gold (Dimension 5) is scored by
`python -m src.eval_expression "<book>" --gold expression_gold.json` — exact + band agreement,
coverage, evidence-support, and confidence calibration, with both sides canonicalized through the
engine's own vocabulary. The per-line emotion distribution (Dimension 4) is inspected with
`python -m src.eval_emotion "<book>"`.

---

## Gold ↔ system field map (quick reference)

| Gold dimension | System artifact | Key fields |
|---|---|---|
| 0 Segmentation | `performance_script.json` | `scenes[].lines[].{line_id, scene_id, segment_type, text}` |
| 1 Structure | `book_structure.json`, `loop3_boundaries.json` | parts/chapters/scenes, titles, boundary confidence + reasons |
| 2 Attribution | line payloads, `loopE_llm_alias_merges.json` | `character`, `speaker_id`, aliases |
| 3 Reduction | `performance_script.json` | `reductions[]`, `attribution_reduction`, `attribution_delivery` |
| 4 Emotion | `performance_script.json` | `emotion`, `emotion_context`, `emotion_pass` |
| 5 Expression | `character_profiles.json`, `performance_script.json` | `emotion_expression_profile`, `expression.{style, intensity}` |
| 6 Production | `tier3/*.json` | sound design, production script, generation prompts |

---

## System Vocabulary Reference

*The authoritative enums live in code; this table is reconciled to them and should be re-synced if
they change. Verify with `python -c "import src.emotion_pass, src.expression_profile"`.*

- **Emotion labels** (`src/emotion_pass.py`): `flat, cheery, sad, angry, violent, tense, fearful`
  (plus `neutral` as a secondary no-signal label; prefer `flat`).
- **Expression styles** (`src/expression_profile.py → SUGGESTED_EXPRESSION_STYLES`): `hidden,
  internalized, suppressed, restrained, subtle, dry, quiet, controlled, measured, open, expressive,
  overt, intense, explosive`.
- **Delivery cues** (from attribution reduction): `whisper, mutter, shout, scream, hiss, snap`.
- **`segment_type`**: `dialogue, narrative`. **`utterance_type`**: `speech, vocalization`.
- Emotion-noun aliases and style aliases are defined in `src/expression_profile.py`
  (`_EMOTION_ALIASES`, `_STYLE_ALIASES`) — use canonical terms and treat aliasing as a safety net,
  not a license for free-form vocabulary.

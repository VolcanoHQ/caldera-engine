# Tier 2 readiness — first-pass confirmation

*Does the Caldera Engine analysis provide the right and complete information to build
Tier 2 (narrator + attributed character voices)? Check via `validation/tier2_readiness.py`.*

## Verdict

**The information MODEL is complete; the attribution DATA QUALITY is the gate.**

A Tier 2 render (`production_mixer.mix_voice_track`, `resolve_line_wavs`) consumes, per
dialogue line: the **character** (→ voice selection), the **emotion** (→ synthesis), and
the **reduced text** (no tags read aloud) — plus a **castable roster**. The analysis
provides every one of these fields, on every line, for every book checked:

- emotion coverage: **100%** (e.g. 283/283 on A Scandal)
- text coverage: **100%**, with redundant attribution tags now silenced (Fix landed this pass)
- roster + voice affinity (marketplace query per character): present

**Peter Rabbit is 100% Tier-2-ready end to end** (6/6 dialogue attributed, 3 castable
characters, tags silenced). So the pipeline and schema are right.

## What blocks a clean Tier 2 on the rest — all attribution (LLM enrichment) quality, not schema

| Book | Dialogue attributed | Roster issue |
|---|---|---|
| Peter Rabbit | **100%** ✅ | clean (3 characters) |
| The Red-Headed League | 92% (18 Narrator) | duplicates: Mr. Wilson=Jabez Wilson, Assistant=Spaulding; false: "China" |
| A Scandal in Bohemia | 82% (52 Narrator) | **King fragmented into Bohemia(35)+King(25)+Ormstein(7)**; Mr. Sherlock=Holmes; false: "Slim youth in an ulster" |
| Gift of the Magi | **0%** ❌ | no roster — never enriched |
| Alice (ch.1) | **0%** ❌ | no roster — never enriched |

Two enrichment-layer gaps, both confirmed:

1. **Attribution coverage** — 8–18% of dialogue on the enriched Sherlock stories is still
   `Narrator` (unattributed); un-enriched books are 0%. Those lines would be read by the
   narrator voice instead of the character.
2. **Alias resolution failed** — `loopE_llm_alias_merges.json` is empty (`[]`) for the
   Sherlock books, so one character appears under several names (the King = Bohemia =
   Ormstein = 67 lines across 3 names). Casting would give one character three different
   voices. A few descriptive non-names also leak in ("Slim youth in an ulster").

## Bottom line

The engine gives Tier 2 the *right kinds* of information and *all* the fields it needs —
schema/pipeline are complete. To ship a *polished* multi-voice cast, the **attribution
pipeline needs higher coverage and working alias-merge** (both LLM enrichment tasks). These
can't be exercised offline; they need a (re-)enrichment run with the current pipeline.

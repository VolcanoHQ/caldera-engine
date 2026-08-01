# Caldera Engine — Canonical Character Registry (Design & ROI)

*Decision this doc supports: whether to evolve MemPalace into a canonical,
cross-manuscript character registry, and whether the payoff to **current** engine
work justifies the build. Leads with impact so the cost/benefit is judgeable.*

---

## The problem (measured, not hypothetical)

Characters are **fragmented** because there is no canonical entity behind attributed
names. Verified this session:

- Aliases are tracked only **per book** (`confirmed_merges` keyed by `book_filename`) and
  are **empty in practice** — `loopE_llm_alias_merges.json` is `[]` for the Sherlock books.
- MemPalace `drawers` are keyed by **bare `character_name`** (global, collision-prone);
  `rooms`/`wings` carry **no `work_id`**.

Concrete cost, from the Tier 2 validation:

- **A Scandal in Bohemia**: the King is split into **Bohemia (35 lines) + King (25) +
  Ormstein (7) = 67 lines under 3 names**; "Mr. Sherlock" duplicates "Holmes". The roster
  reports **10 characters for a real cast of ~5**.
- **The Red-Headed League**: Mr. Wilson = Jabez Wilson; Assistant = Spaulding; "China" is a
  false speaker.

Effect on the engine **today**: a **Tier 2 render casts one character as several different
voices** (the King would get 3), and a character's profile/arc is split across their aliases.

---

## Impact on CURRENT engine activities (the ROI)

This is not only future product — it fixes work the engine does now:

| Activity | Today (broken) | With the registry |
|---|---|---|
| **Tier 2 casting** (flagship) | King → 3 voices, Holmes → 2 | one canonical character → one voice; roster = real cast |
| **L7 character profiles** | arc/appearances split across aliases | one profile per character, arc merged |
| **Character Timeline** (Dashboard, just built) | lists Bohemia / King / Ormstein separately | shows the King with all his scenes |
| **Correction loop** | per-book, ad-hoc alias rules | durable canonical merges, reused + queryable |
| **Voice consistency** | shared drawers collide on name | `character_id` → stable voice, series-safe |
| **MemPalace hygiene** | no work scoping; cross-book pollution | principled entities, rebuildable index |

**All three tiers depend on canonical characters** — each consumes per-character data, so
fragmentation (King = Bohemia = Ormstein) degrades each one:

- **Tier 1 (single narrator, character-aware):** the expression profile and cross-scene
  continuity are keyed by character. Split aliases mean the King's restrained-anger delivery
  and his residual-state carryover **don't apply** to his "Ormstein"/"Bohemia" lines — so even
  one narrator performs the same character inconsistently.
- **Tier 2 (cast):** casting dedup — the King gets **one** voice instead of three.
- **Tier 3 (full production):** the most to gain — character sheets/visual profiles, the
  dialogue director, and the marketplace casting bridge all key on character. Fragmentation
  means **3 King visuals**, split direction, split emotional arc.

Impact scales with how much per-character data a tier uses (Tier 3 > Tier 2 > Tier 1), but it
is **positive for all three** — this is not a Tier-2-only fix.

---

## Design

A **canonical registry built as a DERIVED INDEX** over the per-book artifacts. The per-book
pipeline files stay the source of truth; the registry is **rebuildable from scratch** — this
avoids a two-sources-of-truth problem and keeps the deterministic file pipeline and the
render path intact. **Extend** MemPalace (reuse its SQLite); do **not** remove the drawer/room
tables the renderer depends on.

New tables (SQLite, in MemPalace):
- `works(work_id, title, source_hash, consent)`
- `characters(character_id, canonical_name, work_id)` — the entity
- `aliases(alias, character_id)` — Ormstein/King/Bohemia → one `character_id`
- `appearances(character_id, scene_id, chapter_id, work_id)` — profile ↔ scene ↔ chapter ↔ book
- `voice_assignments(character_id, voice_ref)` — casting / Actor-Marketplace hook

**Population — reuse what exists:**
- `build_character_profiles` / `consolidate_profiles` already assemble per-book characters +
  the appearance index → they become the registry's **writer**.
- Alias → canonical resolution comes from two sources that are **already wired**:
  `save_confirmed_merge` / `/api/confirm_merge` (human curation in the console) and the
  automated `loopE_llm_alias_merges` when enrichment produces it.
- Resolution order at read: attributed name → `aliases` → `character_id`.

---

## Sequencing (value-first)

1. **Registry schema + writer** from `build_character_profiles` — within-work canonical
   characters, no merges yet. Additive, low risk.
2. **Console alias-curation** wired through the existing `confirm_merge` → registry. This
   delivers the **Tier 2 dedup immediately via human merges — no LLM needed**.
3. **Render + timeline read the registry** (Tier 2 casts canonical characters; timeline shows
   them).
4. **Improve LLM alias-merge** to auto-populate and cut manual curation.
5. **Cross-work identity** as an explicit, curated, opt-in link (deferred; gated on IP/consent).

---

## Cost / risk / mitigation

- **Effort:** moderate — a schema, a writer (mostly reusing `build_character_profiles`), a
  console panel, and read-path swaps. **No new infrastructure** (reuse SQLite).
- **Risk:** breaking the render path (MemPalace drawers/rooms). **Mitigation:** additive
  tables; registry is derived/rebuildable; render resolves `character_id` but falls back to
  name.
- **Honest dependency:** the registry **stores** merges; it does not **produce** them. Phase
  1–2 value comes from **human curation (viable now)**; auto-population needs the LLM
  alias-merge to actually work (currently empty). The DB is necessary infrastructure whose
  payoff is unlocked by the merge *decisions*.
- **IP/consent:** cross-work linking and any sharing re-raise the copyright gate; within-work
  + local is unaffected.

---

## Verdict — is it worth it?

**Yes.** Character fragmentation degrades **all three tiers** — Tier 1's character-aware
delivery + continuity, Tier 2's casting, and Tier 3's visuals + production direction — so the
case does not hinge on any single tier. Because Tiers 1, 2, and 3 are all priorities, a fix
that makes every one of them more correct is high-leverage, not niche. And **phases 1–3 deliver
it with human curation alone (no LLM dependency)** while laying the exact backbone the Actor
Marketplace / character-database vision needs — one build, several payoffs.

The only thing that would lower priority is if per-character fidelity across all tiers were
deemed "good enough" as-is — but the measured defects (one character → 3 voices / 3 visuals /
split delivery) are real quality regressions, not cosmetic. Recommend building **phases 1–3
now**, deferring 4–5 (LLM auto-merge, cross-work identity).

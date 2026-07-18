# Firespeaker AI Roster & System Prompts

*Authored 2026-07-04. The authoritative accounting of every AI in the pipeline: how many,
what each one does, its system prompt, provider policy, and validation. If a prompt in code
drifts from this document, the code is right and this document must be updated.*

Every LLM call is identified by its `task_name`, which appears in `data/llm_call_audit.jsonl`
for auditing. Every text-analysis AI follows the same contract:
**structured input → LLM → schema validation (Pydantic) → programmatic grounding check → storage.**
Prompts are suggestions; validation is enforcement. Nothing an LLM outputs reaches audio or
storage without passing a check no model output can override.

---

## The count, by output tier

Using the production taxonomy (Tier 1 audiobook = single narrator; Tier 2 = narrator +
character voices; Tier 3 = full Graphic-Audio production):

| Output tier | Text-analysis AIs required | Generation models | LLM calls for a 7-scene book |
|---|---|---|---|
| **Tier 1** | **0 always-on** + up to 4 loop-gate AIs that fire only on deterministic-pass failure (see next section) | XTTS-v2 (1 voice) | 0 on clean manuscripts |
| **Tier 2** | **+3** — Attribution, Alias Resolver, Clean-Check (advisory) | XTTS-v2 (multi-voice) | ~12 (4 attribution + 7 clean-check + 1 alias) |
| **Tier 3** | **+4 crew** — Spotter, Music Director, Sound Designer, Dialogue Director | + MusicGen (music), AudioLDM (SFX/ambience) | ~40 total (~1 + 6 × scenes) |

Provider chain for all text AIs: **Groq Llama-3.1-8B (volume) → Gemini Flash (quality reserve)
→ local Ollama (offline fallback)** — except where a task is gated Gemini-only (marked below),
and except **pro-plan runs (T2-4)**: a project on the paid plan prepends a `gemini_paid` lane
(own key via `GEMINI_API_KEY_PAID`, paid-tier RPM, no daily budget) that rides the identical
call path and falls through to the free chain on any failure; without a paid key, pro equals
free. The paid lane satisfies Gemini-only task gates — it IS Gemini, same quality class.
Also
because a wrong answer there is worse than no answer. Rate limiting, daily quota tracking, and
cooldowns live in `src/llm_client.py`; per-task provider gates are set at each call site.

---

## Tier 1 loop-gate AIs (deterministic-first; AI engages only on gate failure)

Tier 1's structural loops are deterministic and stay that way — measured at 100% structural
accuracy across Peter Rabbit (7/7 scenes vs gold), Alice (10/10 chapters), and A Scandal in
Bohemia (3/3 sections) with zero LLM calls. An always-on AI per loop would add cost and
nondeterminism where regex is already perfect, and would break Tier 1's zero-cost/offline
contract. Instead, **each loop has a specified gate AI that fires only when the deterministic
pass signals failure** — so the count per loop exists, but the expected call count on clean
manuscripts is zero.

| Gate AI | Loop | Trigger condition (when the AI engages) | Status |
|---|---|---|---|
| **G1: Part Verifier** (`tier1_part_verification`) | Loop 1 (parts) | >20 parts (regex runaway — measured: Les Misérables's nested VOLUME/BOOK headings match as 92 siblings) or 1 part in a >300K-char book | **Implemented 2026-07-11.** LLM identifies the TOP-most division level and lists heading lines verbatim; a prefix pattern derived from grounded examples re-splits book-wide; ≥2000-char spacing and ≥500-char blocks enforced; deterministic parts stand on failure |
| **G2: Chapter Verifier** (`tier1_chapter_verification`) | Loop 2 (chapters) | A single chapter >80K chars, or one "chapter" >8× the median size (missed boundaries) | **Implemented 2026-07-07.** LLM lists heading lines VERBATIM from head+mid samples; boundaries grounded by exact search; <500-char gaps and <200-char blocks rejected; deterministic chapters stand on failure |
| **G3: Filler / Clean-up** | Front-matter & noise | Always available (advisory); auto-triggered when `ClutterScrubber` heuristics signal leakage | **Exists — this is AI-3 (Clean-Check)**, Gemini-gated, advisory-only |
| **G4: Director's Scene Segmenter** (`tier1_scene_segmentation`) | Loop 3 (scenes) | No explicit typographic markers produced the boundaries (measured: 100% of the corpus — a 23-book survey found 0 marker-derived scenes) | **Implemented 2026-07-04.** Director's definition (a scene breaks on LOCATION/TIME/PRESENT-CAST continuity break), windowed ~9K chars, every boundary grounded to a verbatim snippet, <400-char scenes rejected, deterministic scenes stand on failure. Measured: Peter Rabbit 8 scenes vs 7 gold (3 exact + 1 near boundary match; over-segmentation of the escape mirrors the human Tier 3 gold's own micro-scenes; the honest regex baseline yields only 3 scenes now that its hardcoded Peter-Rabbit phrases are removed). Alice: 10 whole-chapter blobs → 46 semantic scenes with correct ch1 dramatic structure. |
| **G5: Line Parser Verifier** | Loop 4 (narration/dialogue split) | Not planned: the quote-parser measured 100% segment-type accuracy vs Tier 2 gold; a gate here would have nothing to catch. Revisit only if a corpus text breaks it | Not planned (measured unnecessary) |

Design rule for all gates: the deterministic output is never discarded — a gate AI's output
replaces it only after passing the same schema validation + text-grounding checks as every
other AI, and the deterministic result remains the fallback if the gate AI fails.

---

## Tier 2 analysis AIs (src/tier_1_parser.py, opt-in via --enable-llm-enrichment)

### AI-1: Speaker Attribution (`tier1_attribution`)
- **Cardinality**: 1 call per ~30 dialogue lines (windowed). G4 director scenes run far
  larger than regex scenes — measured: a 136-line scene overflowed Groq's 6K-token
  request cap (413) and timed out the 3B Ollama fallback, leaving the whole scene on
  Tier 1 defaults. Each window carries a scene-text slice local to its own lines;
  windows fail independently (partial enrichment beats none). Measured effect on the
  Case of Identity benchmark: 84.4% → **99.2%** (windowing + AI-12 review together).
- **Provider**: full chain (Groq is the measured workhorse; scored 5/5 on Peter Rabbit gold).
- **Input**: full scene text + book-level character roster + indexed dialogue lines.
- **Output schema**: per line: `speaker`, `emotion`, `confidence`, `utterance_type`
  (speech|vocalization); plus `sfx_cues` (see note below).
- **Validation**: index range checks; speaker locked to nearest roster match; confidence
  clamped; lines under 3 alphanumeric chars excluded (quoted-fragment guard); SFX cues must
  appear verbatim in scene text (anti-hallucination) and are dropped if they duplicate dialogue.
- **System prompt (core)**:

> You are an expert literary analysis AI attributing dialogue lines to speakers for an
> audiobook engine. Below is the full text of one scene (for context), a candidate character
> roster (heuristically extracted from the WHOLE BOOK, so it may include characters named in
> earlier scenes but only referenced by pronoun or epithet here), and a list of dialogue lines
> from this scene, each tagged with its index number. Use the surrounding narrative to infer
> who is speaking — an action or attribution tag ("Peter sneezed", "she gave a dose of it to
> Peter") immediately before or after a dialogue line is strong evidence of who that line
> belongs to. If the scene refers to a speaker only indirectly (e.g. "his mother"), map that
> reference to the matching named character from the roster rather than answering "Narrator".
>
> CRITICAL ATTRIBUTION RULES:
> - A name, title, or honorific appearing INSIDE a quoted line usually identifies the person
>   being ADDRESSED (the listener), NOT the speaker.
> - In a conversation between two characters, consecutive dialogue lines usually ALTERNATE.
>   A character almost never replies to their own line.
> - Some roster candidates may be places or objects. NEVER attribute dialogue to a place.
> - utterance_type: "speech" for spoken words; "vocalization" for non-lexical sounds a
>   character produces (a sneeze like "Kertyschoo!", a gasp, a scream) — attributed to the
>   character but performed as sound, not read as words.

*Known overlap to resolve*: this AI currently also emits `sfx_cues` (narrated onomatopoeia).
Since AI-7 (Sound Designer) now covers sound exhaustively, the recommended consolidation is to
remove SFX duty from Attribution and let the Sound Designer own it — one job per AI.

### AI-2: Identity / Alias Resolver (`tier1_alias_merge`)
- **Cardinality**: 1 call per book, after all scenes are attributed.
- **Provider**: **Gemini-only** (measured failure: Groq 8B merged "King" into "Holmes").
  Single-provider tasks wait out rate-limit cooldowns rather than skipping.
- **Input**: every attributed speaker with line counts and sample lines.
- **Output schema**: `groups: [{canonical, aliases[]}]`.
- **Validation**: canonical and aliases must be existing speakers (no invented names); never
  "Narrator"; no alias in two groups; and the **conversation-alternation blocker** — two
  names that trade dialogue ≥2 times are two people talking to each other, and no model
  output can merge them. Results persist to MemPalace `confirmed_merges` (book-scoped).
- **System prompt (core)**:

> You are resolving character identities for an audiobook voice-casting engine. Below are the
> speaker names an attribution pass assigned to dialogue lines in one book, with line counts
> and a sample line each. Some names refer to the SAME person under different labels: titles
> vs personal names ("Majesty" / "King" / "Count Von Kramm" may all be one royal character in
> disguise), partial names ("McGregor" / "Mr. McGregor"), epithets, or misspellings.
> Group ONLY names that clearly refer to the same single person. "canonical" MUST be one of
> the listed speaker names — pick the most complete/specific personal name in the group.
> Never invent a name that is not listed. Do not merge two distinct people just because their
> names are similar. Never include "Narrator" in any group.

*AI-2 addendum (2026-07-07)*: a **deterministic composed-name pre-merge** now runs
before the LLM: if two speakers' name tokens together compose one longer roster name
("Miss Mary" + "Miss Sutherland" = roster's "Miss Mary Sutherland"), they merge without
LLM judgment (still blocker-checked). Also: the alternation blocker now counts only
DIRECTLY ADJACENT dialogue (label-flapping for one person no longer blocks a merge),
and single-provider tasks retry their wait-attempt cycle up to 3 times (measured: one
wait wasn't enough against a hot RPM window).

### AI-12: Continuity Reviewer (`tier1_attribution_review`)
- **Cardinality**: 1 call per 40-dialogue-line window of each scene with ≥6 dialogue
  lines and ≥2 speakers, AFTER attribution (same provider-size constraint as AI-1;
  conversation-flow errors are local, so windowed review loses nothing).
  **Provider**: full chain.
- **Job**: reviews the attributed conversation FLOW — the error classes per-line
  attribution can't see (measured residuals on the 122-line Case of Identity benchmark:
  rapid two-party exchange drift, addressee-inversion survivors, minor characters
  absorbing protagonist lines). Flags lines breaking conversational logic and proposes
  corrections from the scene's existing speaker set only.
- **Validation**: corrected speaker must be in-scene or in-roster; stale-state
  corrections rejected (current_speaker must match); corrections capped at 30% of the
  scene's dialogue lines — a reviewer rewriting everything is itself wrong and the whole
  pass is discarded. Applied corrections get `attribution_method += "+reviewed"`,
  `confidence = 0.75`.

### AI-3: Manuscript Clean-Check (`tier1_cleancheck`)
- **Cardinality**: 1 call per scene. **Advisory only** — flags are written to a sidecar for
  review; nothing is auto-deleted.
- **Provider**: **Gemini-only** (measured failure: Groq 8B flags real story text as noise).
  If Gemini is unavailable, the scene simply goes unchecked — better than garbage flags.
- **Output schema**: `is_clean`, `issues[{issue_type, raw_text, description, suggested_action}]`.
- **System prompt (core)**:

> You are a professional manuscript editor and ingestion auditor for an audiobook production
> engine. Run a "Manuscript Clean Check" on the provided text block. Identify non-narrative
> elements that should be removed or cleaned before speech synthesis: Project Gutenberg
> license headers/footers/metadata; illustration tags; page numbers, headers, transcriber
> notes; extraneous formatting noise (e.g. raw underscores representing italics).

---

## Tier 3 production crew (src/scene_director.py)

The crew mirrors a real Graphic-Audio production team. The **Spotter runs first** and makes
no creative decisions — it only marks WHERE opportunities are. Each specialist then works
its own craft against the spotted marks. Rationale for separate focused AIs over one
generalist prompt: (a) each role's instructions are short enough that the 8B workhorse
actually follows them (measured: the earlier combined prompt ignored constraints the split
prompts honor); (b) each output is independently validated, so one role's failure degrades
only its own layer; (c) the audit log attributes quality problems to a specific role.
The acknowledged trade-off: 4 calls/scene instead of 1. If call volume ever matters more
than quality isolation, the crew can be collapsed onto a single stronger model — the schemas
are designed to compose.

### AI-4: Spotting Artist (`tier3_spotting`)
- **Cardinality**: 1 call per scene, always first; its output feeds AIs 5-7.
- **Output schema**: `music_moments[]`, `sound_moments[]`, `delivery_moments[]`, each entry
  `{line_index, source_text (verbatim), opportunity}`.
- **Validation**: line index in range; source_text verbatim in scene (ungrounded moments dropped).
- **System prompt (core)**:

> You are the SPOTTING ARTIST for a Graphic-Audio-style full-cast audiobook production. Your
> only job is scene breakdown: mark WHERE production opportunities exist. You do NOT decide
> what the music or sounds should be — the music director and sound designer do that from
> your marks. Return: music_moments (dramatic beats where music could shift or accent),
> sound_moments (physical actions, environmental sounds, non-verbal creature/character
> sounds implied by the text — never spoken dialogue), delivery_moments (dialogue lines
> whose emotional delivery is non-obvious).

### AI-5: Music Director (`tier3_music_direction`)
- **Cardinality**: 1 call per scene, consuming the Spotter's music_moments.
- **Output schema**: `music {base_mood, style, stingers[{after_line_index, description,
  trigger_text}]}` + `environment {location, time_of_day, weather, physical_confines,
  ambient_noise_level}`.
- **Validation**: stinger anchors in range; trigger_text verbatim in scene.
- **Downstream**: `style`+`base_mood` become the MusicGen bed prompt; stinger descriptions
  become MusicGen stinger prompts; environment feeds ambience prompts and (future) reverb.
- **System prompt (core)**:

> You are the MUSIC DIRECTOR for a Graphic-Audio-style full-cast audiobook production. Your
> craft: score direction and the scene's acoustic setting. The spotting artist has already
> marked the dramatic beats — score against those marks. Ground every choice in the actual
> text. Produce: base_mood, a concrete musical style (instruments, feel), and 0-3 stingers
> for genuinely notable dramatic beats, each with the verbatim trigger text. Also describe
> the scene environment (location, time of day, weather, confines, ambient noise level).
> Leave delivery_notes empty — the dialogue director handles delivery.

### AI-6: Dialogue Director (`tier3_dialogue_direction`)
- **Cardinality**: 1 call per scene with dialogue, consuming the Spotter's delivery_moments.
- **Output schema**: `delivery_notes[{index, note}]`.
- **Validation**: notes only on dialogue-line indices.
- **Downstream**: notes render into the production script; `[pause:X]` markup +
  `line_overrides.json` provide the human-in-the-loop performance editing path.
- **System prompt (core)**:

> You are the DIALOGUE DIRECTOR for a Graphic-Audio-style full-cast audiobook production.
> Your craft: acting direction for voice performers. The spotting artist has marked lines
> whose delivery is non-obvious — direct those first. Notes are short parenthetical acting
> directions a voice actor performs from — pacing, subtext, physical state ("Warm but
> stern", "Breathless, terrified", "Through gritted teeth"). Do not direct narrative lines.

### AI-7: Sound Designer (`tier3_sound_design`)
- **Cardinality**: 1 call per scene, consuming the Spotter's sound_moments.
- **Output schema**: `continuous_ambience[]` + `events[{name, anchor_line_index, source_text,
  category (action|creature|environment), emotional_intent, layers[{component, timing
  (start|overlap|tail), level (prominent|medium|subtle)}]}]`.
- **Validation**: anchor in range; source_text verbatim in scene; events duplicating spoken
  dialogue dropped (speech is never SFX).
- **Downstream**: each layer becomes an AudioLDM generation prompt (creature layers get the
  emotional_intent appended); layers are composited by ffmpeg with timing offsets and level
  gains into ONE sound event, loudness-normalized, placed at its anchor timestamp.
- **System prompt (core)**:

> You are the SOUND DESIGNER for a Graphic-Audio-style full-cast audiobook production. Your
> craft: foley and effects. The spotting artist has marked the sound moments — design
> against those marks first. Think like a foley artist and field recordist: most real sound
> moments are LAYERED composites of 2-4 component sounds. Produce: continuous_ambience
> (1-3 ongoing background components for the scene's setting) and 0-5 composite events.
> Categories: "action" (physical events), "creature" (non-verbal animal/character sounds,
> with the emotion the sound expresses — e.g. sparrows imploring Peter: "urgent,
> encouraging, don't-give-up"), "environment". Each layer is ONE concrete generatable sound
> ("ceramic pot scraping on wood"). Never include spoken dialogue as sound. Every
> source_text must be copied verbatim from the scene — never invent one and never copy an
> example from these instructions.

### AI-8: Dramatist / Adaptation Writer (`tier3_dramatization`)
- **Cardinality**: 1 call per scene, after the Spotter. **The one AI allowed to invent text** —
  under the grounded-dramatization rule: the WHAT must come from the text (every insert cites
  the verbatim narrated event it dramatizes), only the WORDS are generated. All inserts are
  additive (original lines are never altered) and permanently flagged `is_dramatized`, so
  faithful-mode output can drop them wholesale.
- **Fidelity dial** (product lever): `faithful` (pass skipped — nothing invented),
  `enhanced` (performance vocalizations by existing cast only), `full` (invented dialogue +
  minor-cast minting, e.g. Sparrow 1 / Sparrow 2 / Old Mouse from narrated creatures).
- **Output schema**: `inserts[{anchor_line_index, source_text, insert_type
  (dialogue|performance_vocal), character, text, delivery, meaning_note}]` +
  `new_characters[{name, description}]`.
- **Validation**: anchor in range; source_text verbatim in scene; insert count capped;
  new characters only in `full` mode; inserts never replace existing lines.
- **Downstream**: inserts are synthesized in the character's voice (performance vocals are
  performed, not read — the "Kertyschoo!" rule) and spliced into the voice timeline after
  their anchor line; new minor characters are auto-registered with distinct voices.
- **Derived from**: the corrected `HumanProcessed/Tier 3` Peter Rabbit gold (2026-07-04),
  which established grounded dramatization as the Tier 3 standard (~9 dramatized beats:
  eating noises, tummy groan, sob speech, sparrow duo, mouse exchange, sneeze performance,
  distant McGregor yell, weak "Mother...", tea gag). Dramatist quality is scored on **beat
  coverage** (did it stage the same moments?), not word-match.

*AI-8 addendum (2026-07-07)*: **beat-directed fill** (`tier3_beat_dramatization`)
implemented — after the broad passes merge, spotted moments with no insert within
±1 line each get one targeted single-moment call (capped 4/scene). Measured on Peter
Rabbit: 16 → 23 inserts, ~6/9 gold beats (from 4/9). Known new defect class it exposed:
source-text-copied-as-performance (narration returned verbatim as a "vocal") — a
copy-guard is the next validation to add.

*Also specified by the corrected gold, next in build order after the Dramatist:*
*(a) music state machine (bed/cut/resume/transform events mid-scene, replacing static
bed+stingers); (b) sentence-level SFX anchoring (sub-line precision).*

### AI-10: Character Designer (`tier3_character_design`)
- **Cardinality**: 1 call per book. **Provider**: Gemini→Groq (never Ollama).
- **Job**: one paintable visual profile per main character for Chain E image
  consistency — grounded where the text describes them, explicitly flagged
  `inferred` where invented (period/genre-consistent). Profiles stored in
  `tier3/character_profiles.json` + each drawer's `visual_profile` key, and appended
  to every scene-still prompt featuring that character.
- **Known gaps (measured on first run)**: main-cast selection used dialogue-line
  counts, which excluded Peter (the protagonist speaks once!) — must include
  narrative-mention counts; and the evidence grounding accepts any mention snippet,
  not specifically *descriptive* ones, so `inferred` under-reports.

### AI-11: Production QC Critic (`tier3_qc_review`)
- **Cardinality**: 1 call per book, over the assembled tier3 artifacts (music
  directions, sound design, dramatization). **Provider**: Gemini→Groq. **Advisory
  only** — writes `tier3/qc_report.json`, changes nothing.
- **Job**: flag anachronisms vs the bible, register violations, contradictions,
  grating repetition. Every flag must quote the offending artifact text VERBATIM
  (ungrounded flags dropped).
- **Measured on first run**: grounding held (5/5 flags verbatim); the repetition
  flag was genuinely useful; but two "anachronism" claims were factually wrong
  (cello and clarinet are perfectly Victorian) — treat musicology claims from the
  8B fallback as suggestions, never auto-apply.

---

## Generation models (not prompt-driven analysis; consume the crew's output)

| Model | Role | Where | Notes |
|---|---|---|---|
| **XTTS-v2** (coqui-tts) | Character voices | `voice_synthesizer.py` | Cloned refs or pinned builtin speakers (`xtts_speaker` in MemPalace drawer); `[pause:X]` markup for directed pacing; CPML non-commercial license — revisit before commercial release |
| **MusicGen small** (transformers) | Music beds + stingers | `audio_generation.py` | Loudness-normalized (beds −23 LUFS, stingers −16); cached by prompt |
| **AudioLDM v1** (diffusers) | SFX layers + scene ambience | `audio_generation.py` | v1 deliberately, not v2 (transformers compat); cached by prompt |

Assembly (`production_mixer.py`) is deterministic — no AI: line durations → event timestamps,
sidechain ducking, composite layering, ACX mastering.

## Open questions / recommended next changes

1. **Remove SFX duty from AI-1** (Attribution) now that AI-7 owns sound exhaustively.
2. The Spotter/specialist split costs 4 calls/scene; revisit if a single stronger model
   (Gemini Pro tier, or a larger local model) proves able to hold all constraints at once.
3. Delivery notes are produced but not yet consumed by synthesis (only emotion → pitch/speed
   post-mods today). Wiring notes → XTTS performance is the next quality lever.
4. Scene sub-segmentation for novel-scale chapters (Alice's whole-chapter scenes are too
   coarse for per-scene music/ambience direction).

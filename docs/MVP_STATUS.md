# Caldera Engine — Current State & MVP Kanban

*Snapshot date: 2026-08-16. This is a living status board, not a DoD ledger —
for full Definition-of-Done evidence on any item, see
`docs/Caldera Engine Roadmap & Definitions of Done.md` (the authoritative
per-item ledger). This doc exists to answer one question at a glance: **what's
left before Caldera Engine + Voice Marketplace is a shippable MVP?***

---

## 1. The MVP sentence

> EPUB/DOCX/TXT in → review & correct in the Console → choose a tier → one
> button → chaptered audiobook out — **with narrator voices sourced from a
> real Voice Marketplace where actors can sell their voice and get paid.**

The first half (ingest → render → export) is complete and DoD-verified. The
second half (marketplace as a real, payable product) is the primary gap
between "internal tool" and "MVP" today.

---

## 2. Two products, one story

| Product | Repo | Status |
|---|---|---|
| **Caldera Engine** (was Firespeaker) — manuscript → audiobook pipeline, Console, Express Mode GUI | `VolcanoHQ/caldera-engine` (this repo) | Core pipeline MVP-complete; Express tier-selector bug fixed and root-caused (§3) |
| **Volcano Studios Voice Marketplace** — standalone marketplace for voice actors to list/sell voices & variations | `VolcanoHQ/voice_marketplace` (separate repo, `d:\source\volcano-studios-voice-marketplace`) | Extracted, running locally via Docker (port 8010), talks to this app over HTTP via `src/marketplace_client.py` |

They integrate over a plain HTTP contract (`MARKETPLACE_API_URL` in
`.env.example`) — no shared code, no shared database. Either can be deployed
independently.

---

## 3. Kanban

### ✅ Done

- **Tier 1 MVP pipeline** (M-1/M-2/M-3 in the DoD ledger): one-button render
  job, chaptered M4B export with line-timing manifest, EPUB ingestion.
- **Tier 2/3 production** (attribution, scene director, mixer, media
  generation chains): built and DoD-verified.
- **Console** (Phase 1), **Voice Cloning Studio wizard**, **user accounts +
  auth** (magic link, sessions, `CALDERA_AUTH=on/off`), **per-user projects &
  usage metering**, **paid-lane LLM entitlement**.
- **Scene Mixer (Tier 3 advanced mode)**: full read/write override API +
  Console UI, per-asset regeneration.
- **Rename to The Caldera Engine** + GitHub Pages landing page.
- **Voice Marketplace extracted to a standalone product**: own repo, own
  FastAPI backend, own frontend, Stripe Connect payments (test-mode),
  variations schema (emotional takes + algorithmic pitch/speed variants), real
  sentence-embedding search (replacing keyword simulation). Deployed locally
  via `docker compose` on port 8010, verified end-to-end against this repo's
  `marketplace_client.py`.
- **Per-voice XTTS fine-tuning tier** (premium narrator quality upgrade):
  corpus accumulation across sessions, gated job lifecycle, LRU-cached
  checkpoint loading in the synthesizer, 40 new tests, `.env`-configurable
  (`CALDERA_FINETUNE_*`). Scoped to stay Firespeaker-local — fine-tuned
  checkpoints are never pushed to the marketplace product.
- **GPU synthesis path** (`CALDERA_TTS_DEVICE=auto/cpu/cuda`): 3.4× realtime
  speedup measured, OOM-safe demotion to CPU mid-render.
- **Express Mode tier selector bug** (`3a0a8dc`, `5c9a0d1`, `34fff42`) —
  fully root-caused and fixed across three commits: (1) the Express "Audio
  Synthesis" panel never had a tier-card selector wired to `productionTier`
  at all (Express mode hides the Pro-Mode tab that owned the only radios) —
  added one; (2) the *real* root cause of "nothing responds to clicks" was
  that ~13 JS functions (tier readiness, environment preflight, tier
  selection) were physically located inside the `<style>` tag instead of
  `<script>` — browsers silently discard unrecognized text in `<style>` as
  invalid CSS rather than erroring, so this code never executed; moved it
  into the real `<script>` block; (3) once that code started running, a
  second latent bug surfaced — `esc()` was called in 3 places but never
  defined anywhere in the file — added a minimal HTML-escape helper. A
  proactive scan for other undefined-function references in the script
  turned up nothing else. Confirmed via real browser console errors
  (not just HTML-source inspection, which had given false confidence
  earlier) at each step.

### 🔧 In Progress

- **Local dev environment quirks** (Windows-only, workarounds in place, not
  root-caused upstream): port 8082 unbindable on this machine (using
  `CALDERA_GUI_PORT=8085` instead), console `UnicodeEncodeError` on boot
  banner (using `PYTHONIOENCODING=utf-8`).

### 📋 Backlog — required to call the marketplace tie-in "MVP-real" (not just working locally)

1. **Real Stripe credentials + go-live checklist** for the standalone
   marketplace: currently test-mode placeholders only; needs real Stripe
   Connect account, webhook endpoint on a real reachable URL (no more proxying
   through this app — already decoupled), and a production `.env`.
2. **Public-reachable deployment of both products** (currently both are
   localhost-only): pick hosting (VPS/Docker host/cloud) for the marketplace
   first since it's the one that needs to be reachable by Stripe webhooks and
   by voice-actor sellers; Caldera Engine itself can stay a downloadable/local
   tool longer if desired.
3. **T2-5 · Marketplace identities + payments** (per the DoD ledger, still
   ⬜ OPEN at the *integration* layer): confirm Caldera Engine's own user
   accounts (magic-link auth) can be linked to a marketplace seller/buyer
   identity end-to-end, not just called over HTTP anonymously today.
4. **Hardened auth before any non-local exposure**: `CALDERA_AUTH=on` works,
   but HTTPS + rate limiting called out as still-needed in the ledger before
   the Caldera Engine server itself is safe to expose past localhost.
5. **Marketplace browse/search polish**: current UI is functional but
   minimal; needs real seller onboarding UX passes, buyer trust signals
   (samples, reviews), and a real payout/balance view for sellers.
6. **Fine-tuning tier real-world validation**: built and unit-tested, but
   never run end-to-end against a real GPU + the `coqui-tts` training extras
   (this dev sandbox has neither) — needs a GPU-equipped test pass before
   calling it production-ready.
7. **T2-6 remainder**: per-chapter incremental rendering + a real job queue
   beyond today's one-job-per-book detached workers.

### 🧊 Deferred / Icebox (explicitly not MVP-blocking)

Hamlet/play-script parser · Chain F (video) · cross-book series identity
table · full README rewrite.

---

## 4. Architecture snapshot

```
Caldera Engine (this repo)                Volcano Studios Voice Marketplace (separate repo)
├── src/gui_server.py    Express/Pro GUI   ├── FastAPI REST: listings, search,
├── src/render_job.py    one-button render │   variations, onboard, checkout,
├── src/tier_1_parser.py                   │   seller/connect, webhook/stripe
│   .../scene_director.py  tier 2/3 logic  ├── Qdrant vector search (real
├── src/production_mixer.py  audio mixing  │   sentence-transformer embeddings)
├── src/voice_synthesizer.py  XTTS + fine- ├── Stripe Connect (test-mode)
│   tuned checkpoint-aware synthesis       └── Own frontend (browse/onboard/buy)
├── src/voice_finetune.py  premium tier
├── src/marketplace_client.py  <--- HTTP --->  (the only coupling between the two)
├── src/user_db.py  auth, sessions
├── src/project_db.py  per-user projects
└── src/static/*.html  Console + Express/Pro Studio GUI
```

---

## 5. Risks / open questions

- **Windows dev-box quirks** (port reservation, console encoding) are papered
  over with env vars, not fixed at the root — fine for one dev machine, worth
  revisiting if the team grows or a CI runner is added on Windows.
- **Stripe is test-mode only** everywhere right now — no revenue is possible
  until real keys + a real go-live pass happen.
- **Marketplace and Caldera Engine have separate git histories now** — a
  worktree metadata break during this session revealed a batch of prior
  session work had never actually been committed; that's been repaired and
  pushed, but it's a reminder to `git status`/commit more frequently rather
  than relying on long-lived uncommitted working trees.
- **Verifying JS fixes via HTML-source/curl inspection alone is not
  sufficient** — the tier-selector bug looked fixed twice by that method
  before real browser console errors revealed it wasn't. Always confirm
  JS-touching fixes with an actual browser (or at minimum a JS interpreter
  executing the real runtime path, not just a syntax check).

---

*Source data for this doc: the session's SQL todo ledger (26 todos, 23 done /
3 blocked-and-superseded) cross-referenced with
`docs/Caldera Engine Roadmap & Definitions of Done.md`. Update both as work
lands — this file is meant to be skimmed in under two minutes, the DoD ledger
is where the receipts live.*

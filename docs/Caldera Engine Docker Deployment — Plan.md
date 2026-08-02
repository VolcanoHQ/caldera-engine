# Caldera Engine — Docker Deployment (Plan)

*Decision-support for containerizing the app + pipeline. Grounded in the real
dependency footprint.*

## Why containerize (real wins, not just tidiness)

- **Reproducible, pinned environment.** The ingest crash we just fixed was a
  ChromaDB Rust-binding version mismatch in a local env. A container **pins
  chromadb + torch + coqui-tts** so that class of "works on my machine" breakage
  disappears — the runtime fix stays, but the root cause (drifting deps) is gone.
- **One-command deploy + clean restarts** with state on volumes.
- **Portability** to a GPU host or cloud later without re-solving the env.

## The footprint that shapes the image

- **Heavy ML deps:** `torch>=2.0`, `coqui-tts` (XTTS-v2), `transformers` (Bark),
  `chromadb`, `spacy`. → large image, GPU-relevant.
- **System dep:** **ffmpeg** (subprocess calls in `production_mixer` /
  `audio_generation`). Must `apt-get install ffmpeg`.
- **Models (multi-GB, downloaded on first use):** XTTS-v2 (~1.8 GB), spaCy
  `en_core_web_lg` (~560 MB), MusicGen/AudioLDM (GBs, Tier 3 only).
- **Persistent state:** `data/` (corpus, **mempalace** SQLite+chroma, uploads,
  render_jobs — 267 MB) and `scratch/` (render workspace + outputs — 3.1 GB). Must
  survive restarts → volumes.
- **Runtime:** `python -m src.gui_server` on **:8082**; `render_job` spawns
  detached `python -m src.render_job` subprocesses (fine inside one container).
- **Secrets:** LLM keys via env (`.env`) — inject as env vars, never bake.

## Recommended approach (MVP → scale)

**MVP — a single multi-stage image, CPU-first:**
1. **Multi-stage Dockerfile.** Builder stage installs pip deps into a venv; final
   stage = `python:3.11-slim` + `ffmpeg` + the venv. Keeps the image lean.
2. **GPU is available and viable on this host — use it.** Probed the dev machine:
   NVIDIA **RTX A3000 (6 GB)**, WSL2 GPU passthrough (`/usr/lib/wsl/lib/libcuda.so`),
   Docker + **NVIDIA Container Toolkit** all present. So the GPU image is not a
   hypothetical follow-up — it works here today. Build with CUDA-enabled torch
   (`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`);
   the container needs only the CUDA *runtime* (bundled in the wheel) — the driver
   comes from the host via the toolkit. Launch with `--gpus all` (compose:
   `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`).
   Keep a **CPU build arg** as the portable fallback for hosts without a GPU.
   **VRAM caveat:** 6 GB comfortably runs Tier 1/2 (XTTS-v2 ~2–4 GB); Tier 3 stacks
   XTTS + MusicGen + AudioLDM, which can OOM at 6 GB unless models load/offload
   sequentially. Verify in-container with `nvidia-smi` and `torch.cuda.is_available()`.
3. **Models via a mounted cache volume, not baked.** Mount `HF_HOME` / the TTS
   cache as a named volume so the image stays ~2–3 GB instead of 10 GB+, and
   models persist across rebuilds (pre-populate the volume once, or let first run
   download). Baking is the alternative if you want a fully self-contained image
   and accept the size.
4. **Volumes for state:** `data/` and `scratch/` as named volumes so the
   MemPalace DB, uploads, and renders survive `docker restart`.
5. **Config:** `EXPOSE 8082`, `HEALTHCHECK` curling `/`, `CMD ["python","-m","src.gui_server"]`,
   `spacy download en_core_web_lg` at build (or lazy at first run).
6. **`docker-compose.yml`** wiring the volumes + `env_file: .env` + port — the
   one-command local deploy.

**Scale later (not for alpha):** split **web** (gui_server) from **worker**
(render queue) containers sharing the data volume, so long renders don't tie up
the UI container; add a reverse proxy + real auth (`CALDERA_AUTH=on`).

## Decisions to make before I build it

| Decision | Options | Default I'd pick for alpha |
|---|---|---|
| GPU vs CPU | CPU (portable, slow) / CUDA (fast) | **GPU** — host is confirmed capable (RTX A3000 + toolkit); CPU as build-arg fallback |
| Models | bake into image / mounted cache volume | **mounted cache** (lean image) |
| Topology | one container / split web+worker | **one container** for alpha |
| Tiers in scope | Tier 1 only (no MusicGen/AudioLDM) / all tiers | decide by what you'll alpha |

## Risks / notes

- **Image size + first-run downloads** are the main friction; the cache-volume
  strategy contains it.
- **`.dockerignore` is essential** — exclude `data/`, `scratch/`, `.git`,
  `validation/output/`, model caches, or the build context balloons to GBs.
- CPU synthesis of a full book is slow; alpha on short manuscripts is fine.

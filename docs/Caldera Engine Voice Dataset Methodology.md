# Caldera Engine Voice Dataset Methodology

*Authored 2026-07-13. The "separate but connected" lane of Volcano Studios: how a human
voice becomes a production-grade asset. Tooling lives in `src/voice_dataset.py`; the
marketplace consumes its output via `voice_marketplace.onboard_voice`.*

## 1. What a voice dataset is for (three tiers of use)

| Use | Audio needed | When |
|---|---|---|
| **Zero-shot reference set** | 6–30 s of clean, expressive speech | Immediately — XTTS-v2 conditions on it directly; this is what the marketplace binds to characters today |
| **Emotional reference bank** | 1–3 clips *per emotion* (5–15 s each) | Near-term — fills MemPalace's `emotional_references` table so "angry McGregor" conditions on *your* angry read instead of a pitch modifier |
| **Fine-tuning corpus** | 30–60+ min, transcript-aligned | Premium tier (implemented, §8) — real per-voice XTTS-v2 fine-tuning for maximum fidelity; grow past ~15 min via repeated `intake --transcripts` sessions |

Design rule: **record once, serve all three.** The session script below produces a
dataset that satisfies the zero-shot need on day one and accumulates toward the
fine-tuning corpus without re-recording.

## 2. Recording protocol

- **Room**: the quietest you have; soft furnishings beat bare walls. No fans/HVAC/PC
  in mic range. Record 10 s of *room tone* (silence) first — the intake tool uses it
  to estimate your noise floor.
- **Gear**: any decent USB/XLR mic > headset mic > phone; all are usable. Disable
  ALL processing: no AGC, no noise suppression, no "voice isolation" (they smear the
  spectrum XTTS conditions on). Phone voice-memo apps: use a "lossless/WAV" setting
  if available.
- **Format**: WAV, mono preferred, **48 kHz / 24-bit** ideal — the tool downsamples
  correctly; upsampling a bad recording is impossible.
- **Technique**: 15–20 cm from the mic, slightly off-axis (reduces plosives), steady
  distance. If you clip (meters hitting red), step back — clipping is the one defect
  the pipeline cannot repair.
- **Takes**: one WAV per prompt, named by the prompt id (`A01.wav`, `B03.wav`,
  `C02_angry.wav`). A single long take per block also works — the tool splits on
  silence and walks you through assignment.

## 3. Session script design (why these blocks)

`voice_dataset.py init` generates a personalized prompt sheet with three blocks:

- **Block A — phonetic coverage** (8 prompts, ~2 min): the Rainbow Passage split into
  breath-group chunks plus Harvard-list sentences. Together they cover the full
  English phoneme inventory and common clusters — this is what makes the corpus
  fine-tune-ready rather than just "some speech".
- **Block B — domain match** (6 prompts, ~3 min): narrative prose read at audiobook
  pace, drawn from the public-domain corpus this studio actually produces (Peter
  Rabbit, Sherlock Holmes narration). Zero-shot cloning transfers *register*: a voice
  sampled reading narration performs narration better than one sampled chatting.
- **Block C — emotional and character range** (10 prompts, ~4 min): a fixed carrier
  sentence read in each target emotion (Neutral, Joy, Sadness, Anger, Fear, Surprise,
  Whisper, Projected/Shout) plus two free character reads. Each clip is labeled with
  its emotion in the manifest — these seed the emotional reference bank.

Total session: **under 15 minutes of speech**, which yields a strong zero-shot set,
a complete emotion bank, and ~10% of a fine-tuning corpus. Repeat Block B with new
passages in later sessions to grow the corpus; Blocks A and C rarely need re-recording.

## 4. Intake pipeline (enforcement over trust, as everywhere)

`voice_dataset.py intake` validates every clip mechanically — no self-reported quality:

1. **Decode & inspect**: sample rate ≥ 22.05 kHz, duration 2–20 s per clip.
2. **Clipping**: fraction of samples at ≥ 99% full scale must be < 0.1%.
3. **Silence / level**: RMS above −45 dBFS (not a dead mic), below −6 dBFS (not slammed).
4. **SNR estimate**: 10th-percentile frame energy (noise floor) vs. median speech
   frame energy; flag < 20 dB, reject < 12 dB.
5. **Normalization**: resample to 24 kHz mono, loudness-normalize to −23 LUFS
   (matching the pipeline's bed standard), optional denoise via the existing
   `voice_synthesizer.denoise_audio_file` for flagged-but-usable clips.

Every clip gets a PASS / FLAG / REJECT verdict with the measured numbers; the report
is written next to the dataset. FLAG clips are usable for zero-shot but excluded from
the fine-tuning manifest.

## 5. Dataset layout & manifest

```
data/voice_datasets/{name}/
  recording_script.md      # the personalized prompt sheet
  raw/                     # untouched originals (never modified)
  clean/                   # 24 kHz mono normalized clips, one per prompt
  refs/
    reference_mono.wav     # best-N concatenation for zero-shot conditioning
    emotion_{label}.wav    # per-emotion references
  manifest.csv             # id|transcript|emotion|duration_s|snr_db|verdict
  qc_report.json           # per-clip measurements
  consent.json             # statement, date, sha256 of every raw file
```

`manifest.csv` uses the LJSpeech-style layout every open fine-tuning recipe
(Coqui, F5-TTS, StyleTTS2) ingests, with transcripts taken from the prompt sheet —
scripted recording means no ASR pass is needed, though one can be added as a
verification gate later.

## 6. Consent & provenance (non-negotiable)

`consent.json` records the seller's consent statement, timestamp, and a SHA-256 of
every raw file — the dataset is cryptographically tied to what was actually recorded.
The marketplace's `onboard_voice` already refuses listings without consent; the
dataset tool writes the record it checks. A voice can be revoked by its owner:
listings are removable, and the license ledger records who bought what while it was
live (the licensing story mirrors the [[voice-marketplace]] lane in the roadmap doc).

## 7. Marketplace hand-off

`voice_dataset.py onboard` bridges to the existing lane: it feeds the reference set
to `voice_marketplace.onboard_voice` (which re-validates and assembles its own
cloning reference), attaches the dataset path to the listing for future fine-tuning,
and the voice becomes searchable/castable like any other. From there the normal flow
applies: search → license → `cast_character` → MemPalace drawer → every synthesis.

## 8. Fine-tuning tier (implemented)

Once a dataset's accumulated, non-REJECT audio crosses a minimum duration
(`CALDERA_FINETUNE_MIN_MINUTES`, default 30 minutes), a real per-speaker XTTS-v2
GPT-decoder fine-tune can be triggered — a genuinely separate model checkpoint per
voice, rather than the zero-shot conditioning every other tier uses.

**Growing the corpus past ~15 minutes.** The fixed 26-prompt script alone only
covers phonetic/emotional range, not duration. `voice_dataset.py intake` accepts a
repeatable `--transcripts <sidecar.json>` argument mapping extra freeform clip ids
to their transcript text (`{"EXT_001": "some line the actor read", ...}` or
`{"EXT_001": {"emotion": "Joy", "transcript": "..."}}`); running `intake` multiple
times (once per recording session) merges into the same `qc_report.json` instead of
overwriting it, so a voice actor can do several additional read-aloud sessions
(e.g. reading audiobook chapters aloud) to reach the 30-60+ minute threshold.

**Triggering and monitoring a job:**
```
python -m src.voice_dataset finetune --name <dataset> [--epochs N] [--batch-size N]
python -m src.voice_dataset finetune-status --name <dataset>
python -m src.voice_dataset cancel-finetune --name <dataset>
```
`finetune` validates corpus readiness and launches `scripts/finetune_xtts_voice.py`
as a **detached subprocess** (never in-process — this is a potentially multi-hour,
GPU-heavy job); job state (`queued`/`running`/`ready`/`failed`) is persisted to
`data/voice_datasets/<name>/finetune/status.json`, which self-heals to `failed` if
the recorded process dies without ever writing a terminal state. See
`src/voice_finetune.py` for the full lifecycle implementation.

Once a job reaches `ready`, pin the resulting checkpoint to a character with
`MemPalace.set_finetuned_checkpoint(character_name, checkpoint_dir)`;
`voice_synthesizer.py`'s `synthesize_line` then loads that checkpoint (cached,
LRU-bounded via `CALDERA_FINETUNE_MODEL_CACHE_SIZE`) instead of the shared
zero-shot model for that character, falling back safely to zero-shot if the
checkpoint is ever missing or fails to load.

**Environment requirements.** Real training needs a CUDA-capable GPU and the
`coqui-tts` training extras (`trainer`, `TTS.tts.layers.xtts.trainer.gpt_trainer`) —
unavailable in most dev sandboxes. `scripts/finetune_xtts_voice.py` fails fast with a
clear `status.json` error (not a silent hang or an opaque stack trace) when either is
missing, so the gap is visible to whoever is polling `finetune-status`.

**Scope decision (flagged, not the marketplace's concern).** Fine-tuned checkpoints
are large (hundreds of MB), GPU-architecture-specific binary artifacts — not audio a
buyer could preview — and are **not** pushed to the standalone Volcano Studios Voice
Marketplace product. Fine-tuning is Firespeaker-local production infrastructure: a
studio that already owns/licensed enough of a voice's audio can build a fine-tuned
model for its own local synthesis pipeline. Selling "fine-tuned tier access" as a
product would need a different concept entirely (e.g. hosted rendering credits, not
a checkpoint transfer) and is out of scope here.

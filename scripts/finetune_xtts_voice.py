#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone XTTS-v2 per-speaker fine-tuning job.

Invoked as a subprocess by src/voice_finetune.py::start_finetune_job() (never
imported/run in-process -- this is a potentially multi-hour, GPU-heavy
training run and must not block the caller). Reads a voice_dataset.py-built
dataset (manifest.csv + clean/ wavs), fine-tunes the XTTS-v2 GPT decoder on
it using Coqui's official recipe, and writes progress/results to a
status.json the caller polls (see src/voice_finetune.py::get_finetune_status).

This follows the same graceful-optional-dependency pattern as the rest of
the codebase (voice_synthesizer.py's HAS_TORCH, voice_marketplace.py's
HAS_TEXT_EMBEDDING_MODEL): torch + the `coqui-tts` training extras
(`trainer`, `TTS.tts.layers.xtts.trainer.gpt_trainer`) are required to
actually train, but this script fails with a clear, actionable status.json
entry (not a stack trace to a caller who can't see it) when they're missing
or no GPU is available -- exactly the situation in most dev sandboxes,
where this path is expected to be exercised via mocks/tests only, never a
real run.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _write_status(status_path: str, status: Dict[str, Any]) -> None:
    status["updated_at"] = time.time()
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, status_path)  # atomic on POSIX and Windows -- a
    # concurrent status read never sees a half-written file.


def _load_manifest(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """[(clean_wav_path, transcript, emotion)] for every non-REJECT clip,
    read from voice_dataset.py's build-step manifest.csv."""
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.csv in {dataset_dir} -- run `voice_dataset build` first.")
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            clean_wav = os.path.join(dataset_dir, "clean", f"{row['id']}.wav")
            if not os.path.exists(clean_wav):
                continue
            rows.append((clean_wav, row["transcript"], row.get("emotion", "Neutral")))
    return rows


def _build_ljspeech_metadata(rows: List[Tuple[str, str, str]], speaker_name: str, out_dir: str) -> str:
    """Coqui's XTTS fine-tuning recipe expects an LJSpeech-formatted
    metadata.csv (audio_file|text|speaker_name, no header) alongside the
    wavs it references. Writes it under out_dir/metadata.csv."""
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for wav_path, transcript, _emotion in rows:
            writer.writerow([os.path.abspath(wav_path), transcript, speaker_name])
    return metadata_path


def run_finetune(name: str, dataset_dir: str, output_dir: str, status_path: str,
                  epochs: int, batch_size: int) -> None:
    status: Dict[str, Any] = {"state": "running", "name": name, "started_training_at": time.time()}
    _write_status(status_path, status)

    try:
        # Imports deferred and guarded: these are heavy (torch + coqui-tts's
        # training extras) and only needed for a real training run, exactly
        # like HAS_TORCH/HAS_TEXT_EMBEDDING_MODEL gating elsewhere.
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "torch is not installed. Fine-tuning requires the full ML stack "
                "(pip install torch, and coqui-tts[training] or the underlying "
                "`trainer` + `TTS.tts.layers.xtts.trainer` extras) on a machine "
                "with a CUDA-capable GPU. This is expected to be unavailable in "
                "most dev sandboxes -- deploy this job to a GPU-equipped worker."
            ) from e

        try:
            from trainer import Trainer, TrainerArgs
            from TTS.tts.layers.xtts.trainer.gpt_trainer import (
                GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig,
            )
            from TTS.tts.datasets import load_tts_samples
            from TTS.config.shared_configs import BaseDatasetConfig
        except ImportError as e:
            raise RuntimeError(
                "coqui-tts's XTTS fine-tuning trainer modules are not available "
                "in this environment (trainer / TTS.tts.layers.xtts.trainer.gpt_trainer). "
                "Install the training extras on a GPU worker before running this job."
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA-capable GPU detected. XTTS-v2 GPT-decoder fine-tuning is "
                "not practical on CPU (would take days, not hours) -- run this job "
                "on a GPU-equipped machine."
            )

        rows = _load_manifest(dataset_dir)
        if not rows:
            raise RuntimeError(f"No usable clips found in {dataset_dir}/manifest.csv.")

        metadata_path = _build_ljspeech_metadata(rows, speaker_name=name, out_dir=output_dir)
        dataset_config = BaseDatasetConfig(
            formatter="ljspeech", dataset_name=name,
            path=os.path.dirname(metadata_path), meta_file_train=os.path.basename(metadata_path),
            language="en",
        )

        # XTTS base checkpoint: Coqui's recipe fine-tunes the released XTTS-v2
        # GPT decoder rather than training from scratch. TTS.api's TTS class
        # (used for zero-shot inference in voice_synthesizer.py) already
        # downloads/caches this checkpoint on first use; the trainer recipe
        # re-uses the same cache directory convention.
        model_args = GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=66150,
            max_wav_length=255995,
            max_text_length=200,
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
        config = GPTTrainerConfig(
            output_path=output_dir,
            model_args=model_args,
            run_name=f"xtts_finetune_{name}",
            audio=audio_config,
            batch_size=batch_size,
            eval_batch_size=max(1, batch_size // 2),
            num_loader_workers=2,
            epochs=epochs,
            print_step=25,
            save_step=250,
            save_n_checkpoints=1,
            save_checkpoints=True,
            optimizer="AdamW",
            lr=5e-6,
            datasets=[dataset_config],
        )

        train_samples, eval_samples = load_tts_samples(
            [dataset_config], eval_split=True, eval_split_max_size=min(20, max(1, len(rows) // 10)),
            eval_split_size=0.1,
        )

        model = GPTTrainer.init_from_config(config)
        trainer = Trainer(
            TrainerArgs(restore_path=None, skip_train_epoch=False),
            config, output_path=output_dir,
            model=model, train_samples=train_samples, eval_samples=eval_samples,
        )
        trainer.fit()

        # Coqui's Trainer writes best_model.pth (+ config.json) under its own
        # run subdirectory; surface the final artifact location plainly so
        # voice_synthesizer.py's checkpoint loader has one predictable path.
        checkpoint_dir = trainer.output_path
        status.update({
            "state": "ready",
            "checkpoint_dir": checkpoint_dir,
            "completed_at": time.time(),
            "error": None,
        })
        _write_status(status_path, status)

    except Exception as e:
        status.update({
            "state": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "failed_at": time.time(),
        })
        _write_status(status_path, status)
        # Also raise so the subprocess's own exit code / log file (captured
        # by voice_finetune.py's Popen stdout=log_file redirect) show the
        # failure clearly for anyone tailing train.log directly.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XTTS-v2 for one voice-actor dataset.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_finetune(
        name=args.name, dataset_dir=args.dataset_dir, output_dir=args.output_dir,
        status_path=args.status_path, epochs=args.epochs, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

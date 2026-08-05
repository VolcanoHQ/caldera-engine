#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Per-voice XTTS fine-tuning: job lifecycle + corpus readiness gate.

src/voice_dataset.py's "build" step produces an 8-20s zero-shot reference
(good enough for XTTS-v2 to condition on instantly -- see voice_synthesizer.py)
from as little as the ~15-minute guided recording script. That is the
default/free tier and needs nothing else.

This module adds an optional *premium* tier on top: once a voice actor's
accumulated, transcript-aligned recordings for a dataset cross a minimum
total duration (default 30 minutes, CALDERA_FINETUNE_MIN_MINUTES), a real
per-speaker XTTS-v2 GPT-decoder fine-tune can be triggered. Fine-tuning is a
long-running (potentially multi-hour), GPU-heavy, separate-process job --
this module never trains in-process. It:
  - Gates on corpus size (is_finetune_ready).
  - Launches scripts/finetune_xtts_voice.py as a detached subprocess.
  - Persists/reads a small status.json job record (queued/running/ready/failed)
    so callers (voice_dataset.py CLI, a future GUI) can poll progress without
    blocking on the training run itself.
  - Self-heals status if the recorded pid died without updating the file
    (e.g. the process was killed, or the machine rebooted mid-training).

Scope decision: fine-tuned checkpoints are large, GPU-architecture-specific
binary artifacts (typically several hundred MB), not a WAV a buyer could
preview. They are NOT pushed to the standalone Volcano Studios Voice
Marketplace product -- that product only ever deals in reference audio.
Fine-tuning is Firespeaker-local production infrastructure: a studio/producer
who owns (recorded, or licensed with enough accumulated audio) a voice can
build a fine-tuned model for their own local synthesis pipeline. If we ever
want to *sell* "fine-tuned tier" access, that would be a new marketplace
product concept (e.g. a hosted rendering credit), not a checkpoint transfer --
out of scope here.
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("VoiceFinetune")

DATASET_ROOT = "data/voice_datasets"

# Coqui's official XTTS fine-tuning recipe recommends 30-60+ minutes of
# clean, transcript-aligned speech per speaker for a solid result; below that
# the GPT decoder tends to overfit/mumble. Configurable since a producer with
# a really strong dataset (e.g. very clean single-speaker audiobook masters)
# may reasonably choose to try earlier.
DEFAULT_MIN_MINUTES = float(os.getenv("CALDERA_FINETUNE_MIN_MINUTES", "30"))
DEFAULT_EPOCHS = int(os.getenv("CALDERA_FINETUNE_EPOCHS", "6"))
DEFAULT_BATCH_SIZE = int(os.getenv("CALDERA_FINETUNE_BATCH_SIZE", "4"))

_VALID_STATES = {"not_started", "queued", "running", "ready", "failed"}


def _dataset_dir(name: str) -> str:
    return os.path.join(DATASET_ROOT, name)


def _finetune_dir(name: str) -> str:
    return os.path.join(_dataset_dir(name), "finetune")


def _status_path(name: str) -> str:
    return os.path.join(_finetune_dir(name), "status.json")


def _log_path(name: str) -> str:
    return os.path.join(_finetune_dir(name), "train.log")


# ----------------------------------------------------
# Corpus readiness
# ----------------------------------------------------

def corpus_duration_minutes(name: str) -> float:
    """Total duration (minutes) of every non-REJECT clip in the dataset's
    manifest.csv (written by voice_dataset.py's `build` step). Used both for
    the zero-shot reference (which only needs ~18s of it) and as the gate for
    whether there's enough audio to attempt a real fine-tune."""
    manifest_path = os.path.join(_dataset_dir(name), "manifest.csv")
    if not os.path.exists(manifest_path):
        return 0.0
    total_seconds = 0.0
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            try:
                total_seconds += float(row.get("duration_s") or 0.0)
            except (TypeError, ValueError):
                continue
    return total_seconds / 60.0


def is_finetune_ready(name: str, min_minutes: float = DEFAULT_MIN_MINUTES) -> Tuple[bool, float, float]:
    """(ready, minutes_available, minutes_required). A dataset must first
    exist (voice_dataset.py init/intake/build already run at least once)."""
    minutes = corpus_duration_minutes(name)
    return minutes >= min_minutes, minutes, min_minutes


# ----------------------------------------------------
# Job status persistence
# ----------------------------------------------------

def _read_status(name: str) -> Dict[str, Any]:
    path = _status_path(name)
    if not os.path.exists(path):
        return {"state": "not_started"}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"state": "not_started"}


def _write_status(name: str, status: Dict[str, Any]) -> None:
    os.makedirs(_finetune_dir(name), exist_ok=True)
    status["updated_at"] = time.time()
    with open(_status_path(name), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def _pid_alive(pid: Optional[int]) -> bool:
    """Cross-platform "is this pid still running" check. os.kill(pid, 0) is
    the standard POSIX idiom but is unreliable on Windows (it does not raise
    for an already-exited pid), so Windows uses OpenProcess/GetExitCodeProcess
    via ctypes instead."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # signal 0: existence check only, doesn't actually kill
        return True
    except OSError:
        return False


def get_finetune_status(name: str) -> Dict[str, Any]:
    """Reads the job's persisted status, self-healing if a previously
    "queued"/"running" job's recorded process is no longer alive (crashed,
    killed, machine rebooted) without ever writing a terminal status."""
    status = _read_status(name)
    if status.get("state") in ("queued", "running") and not _pid_alive(status.get("pid")):
        status["state"] = "failed"
        status["error"] = status.get("error") or "Training process exited unexpectedly (no terminal status was written)."
        _write_status(name, status)
    return status


# ----------------------------------------------------
# Launching a fine-tune job
# ----------------------------------------------------

def start_finetune_job(
    name: str,
    min_minutes: float = DEFAULT_MIN_MINUTES,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    popen_factory: Optional[Callable[..., "subprocess.Popen"]] = None,
) -> Dict[str, Any]:
    """Validates corpus readiness, then launches
    scripts/finetune_xtts_voice.py as a detached subprocess and persists a
    "queued" status record. Raises ValueError if the dataset isn't ready
    (not enough audio, or already training). `popen_factory` is an injection
    seam for tests -- defaults to subprocess.Popen."""
    existing = get_finetune_status(name)
    if existing.get("state") in ("queued", "running"):
        raise ValueError(f"A fine-tune job for '{name}' is already {existing['state']} (started {existing.get('started_at')}).")

    dataset_dir = _dataset_dir(name)
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"No such voice dataset: '{name}'. Run `voice_dataset init/intake/build` first.")

    ready, minutes, required = is_finetune_ready(name, min_minutes)
    if not ready:
        raise ValueError(
            f"Dataset '{name}' has only {minutes:.1f} minute(s) of usable audio; "
            f"fine-tuning needs at least {required:.0f} minute(s). Record more sessions "
            f"and re-run `voice_dataset build` to grow the corpus."
        )

    finetune_dir = _finetune_dir(name)
    os.makedirs(finetune_dir, exist_ok=True)
    checkpoint_dir = os.path.join(finetune_dir, "checkpoint")
    log_path = _log_path(name)

    script_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "scripts", "finetune_xtts_voice.py",
    )
    cmd = [
        sys.executable, script_path,
        "--name", name,
        "--dataset-dir", dataset_dir,
        "--output-dir", checkpoint_dir,
        "--status-path", _status_path(name),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    popen = popen_factory or subprocess.Popen
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=os.getcwd())

    status = {
        "state": "queued",
        "name": name,
        "pid": proc.pid,
        "started_at": time.time(),
        "epochs": epochs,
        "batch_size": batch_size,
        "corpus_minutes": minutes,
        "checkpoint_dir": checkpoint_dir,
        "log_path": log_path,
        "error": None,
    }
    _write_status(name, status)
    logger.info(f"Fine-tune job started for '{name}' (pid={proc.pid}, corpus={minutes:.1f}min, log={log_path})")
    return status


def cancel_finetune_job(name: str) -> bool:
    """Best-effort termination of a queued/running job's process. Leaves the
    status as "failed" (cancelled) rather than deleting the record, so the
    corpus/log history stays around for a retry or a postmortem."""
    status = get_finetune_status(name)
    if status.get("state") not in ("queued", "running"):
        return False
    pid = status.get("pid")
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, 15)  # SIGTERM (Windows: TerminateProcess-equivalent via CTRL, best-effort)
        except OSError as e:
            logger.warning(f"Could not signal fine-tune process {pid} for '{name}': {e}")
    status["state"] = "failed"
    status["error"] = "Cancelled by user."
    _write_status(name, status)
    return True

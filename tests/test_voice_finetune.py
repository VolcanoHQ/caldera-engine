"""Tests for src/voice_finetune.py: corpus-duration gating, fine-tune job
lifecycle (start/status/cancel), and self-healing status when a launched
process dies without ever writing a terminal state.

Training itself is never exercised here (no GPU / coqui-tts training extras
in this environment) -- `start_finetune_job`'s `popen_factory` injection seam
is used to fake process launching so these tests stay hermetic and fast.
"""

import csv
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import src.voice_finetune as vf


class _FakeProc:
    """Stand-in for subprocess.Popen with a controllable/fake pid."""

    def __init__(self, pid=999999):
        self.pid = pid


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A voice_dataset.py-shaped directory with a manifest.csv, rooted under
    a tmp DATASET_ROOT so tests never touch the real data/ tree."""
    root = tmp_path / "voice_datasets"
    monkeypatch.setattr(vf, "DATASET_ROOT", str(root))
    name = "TestVoice"
    d = root / name
    d.mkdir(parents=True)
    return name, d


def _write_manifest(dataset_dir, durations_seconds):
    manifest_path = dataset_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("id|transcript|emotion|duration_s|snr_db|verdict\n")
        for i, dur in enumerate(durations_seconds):
            f.write(f"clip{i}|some transcript|Neutral|{dur}|25.0|PASS\n")


# ----------------------------------------------------
# Corpus readiness
# ----------------------------------------------------

def test_corpus_duration_minutes_zero_when_no_manifest(dataset):
    name, _d = dataset
    assert vf.corpus_duration_minutes(name) == 0.0


def test_corpus_duration_minutes_sums_manifest_durations(dataset):
    name, d = dataset
    _write_manifest(d, [60, 120, 30])  # 210s = 3.5 min
    assert vf.corpus_duration_minutes(name) == pytest.approx(3.5)


def test_is_finetune_ready_false_below_threshold(dataset):
    name, d = dataset
    _write_manifest(d, [60] * 10)  # 10 minutes
    ready, minutes, required = vf.is_finetune_ready(name, min_minutes=30)
    assert ready is False
    assert minutes == pytest.approx(10.0)
    assert required == 30


def test_is_finetune_ready_true_above_threshold(dataset):
    name, d = dataset
    _write_manifest(d, [60] * 1900)  # ~31.67 minutes
    ready, minutes, required = vf.is_finetune_ready(name, min_minutes=30)
    assert ready is True
    assert minutes > 30


# ----------------------------------------------------
# Job lifecycle
# ----------------------------------------------------

def test_start_finetune_job_rejects_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(vf, "DATASET_ROOT", str(tmp_path / "voice_datasets"))
    with pytest.raises(ValueError, match="No such voice dataset"):
        vf.start_finetune_job("Nonexistent", popen_factory=lambda *a, **k: _FakeProc())


def test_start_finetune_job_rejects_insufficient_corpus(dataset):
    name, d = dataset
    _write_manifest(d, [60] * 5)  # 5 minutes, well below default 30
    with pytest.raises(ValueError, match="only .* minute"):
        vf.start_finetune_job(name, popen_factory=lambda *a, **k: _FakeProc())


def test_start_finetune_job_writes_queued_status(dataset):
    name, d = dataset
    _write_manifest(d, [60] * 1900)  # ready
    status = vf.start_finetune_job(name, min_minutes=30, popen_factory=lambda *a, **k: _FakeProc(pid=12345))

    assert status["state"] == "queued"
    assert status["pid"] == 12345
    assert status["name"] == name
    assert os.path.exists(vf._status_path(name))

    with open(vf._status_path(name), encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["state"] == "queued"
    assert persisted["pid"] == 12345


def test_start_finetune_job_passes_expected_cli_args(dataset):
    name, d = dataset
    _write_manifest(d, [60] * 1900)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(pid=42)

    vf.start_finetune_job(name, epochs=8, batch_size=2, popen_factory=fake_popen)
    cmd = captured["cmd"]
    assert "--name" in cmd and name in cmd
    assert "--epochs" in cmd and "8" in cmd
    assert "--batch-size" in cmd and "2" in cmd
    assert any(a.endswith("finetune_xtts_voice.py") for a in cmd)


def test_start_finetune_job_rejects_when_already_queued(dataset):
    name, d = dataset
    _write_manifest(d, [60] * 1900)
    # Use our own pid so the self-healing liveness check in get_finetune_status
    # (called internally by the second start_finetune_job) sees it as alive.
    vf.start_finetune_job(name, popen_factory=lambda *a, **k: _FakeProc(pid=os.getpid()))
    with pytest.raises(ValueError, match="already"):
        vf.start_finetune_job(name, popen_factory=lambda *a, **k: _FakeProc(pid=os.getpid()))


def test_get_finetune_status_not_started_when_no_job(dataset):
    name, _d = dataset
    status = vf.get_finetune_status(name)
    assert status["state"] == "not_started"


def test_get_finetune_status_self_heals_dead_pid(dataset, monkeypatch):
    name, d = dataset
    _write_manifest(d, [60] * 1900)
    vf.start_finetune_job(name, popen_factory=lambda *a, **k: _FakeProc(pid=99999999))

    # Simulate the process having died without ever writing a terminal status.
    monkeypatch.setattr(vf, "_pid_alive", lambda pid: False)
    status = vf.get_finetune_status(name)
    assert status["state"] == "failed"
    assert "unexpectedly" in status["error"]


def test_get_finetune_status_leaves_running_job_alone_if_pid_alive(dataset, monkeypatch):
    name, d = dataset
    _write_manifest(d, [60] * 1900)
    vf.start_finetune_job(name, popen_factory=lambda *a, **k: _FakeProc(pid=1))

    monkeypatch.setattr(vf, "_pid_alive", lambda pid: True)
    status = vf.get_finetune_status(name)
    assert status["state"] == "queued"


def test_cancel_finetune_job_marks_failed_and_returns_true(dataset, monkeypatch):
    name, d = dataset
    _write_manifest(d, [60] * 1900)
    vf.start_finetune_job(name, popen_factory=lambda *a, **k: _FakeProc(pid=1))

    monkeypatch.setattr(vf, "_pid_alive", lambda pid: True)
    killed = {}
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.setdefault("pid", pid))

    cancelled = vf.cancel_finetune_job(name)
    assert cancelled is True
    assert killed["pid"] == 1
    status = vf.get_finetune_status(name)
    assert status["state"] == "failed"
    assert status["error"] == "Cancelled by user."


def test_cancel_finetune_job_returns_false_when_nothing_to_cancel(dataset):
    name, _d = dataset
    assert vf.cancel_finetune_job(name) is False


# ----------------------------------------------------
# Cross-platform pid liveness check
# ----------------------------------------------------

def test_pid_alive_true_for_current_process():
    assert vf._pid_alive(os.getpid()) is True


def test_pid_alive_false_for_bogus_pid():
    assert vf._pid_alive(999999999) is False


def test_pid_alive_false_for_none_or_zero():
    assert vf._pid_alive(None) is False
    assert vf._pid_alive(0) is False

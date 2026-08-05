"""Tests for src/voice_dataset.py's fine-tuning-support additions:
  - the --transcripts sidecar mechanism that lets a dataset accumulate extra
    freeform recording sessions beyond the fixed ~15-minute prompt script (the
    corpus-accumulation gap needed to ever reach the fine-tuning threshold);
  - the finetune / finetune-status / cancel-finetune CLI command wrappers,
    which just delegate to src/voice_finetune.py.

ffmpeg/ffprobe are not available in this dev sandbox, so `_read_wav_mono`
and `_qc_clip` are monkeypatched (same pattern as the rest of the suite) --
these tests exercise the intake/merge bookkeeping, not the audio QC math.
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src import voice_dataset


@pytest.fixture
def dataset_root(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_dataset, "DATASET_ROOT", str(tmp_path / "voice_datasets"))
    return tmp_path


def _init_dataset(name):
    voice_dataset.cmd_init(name, name)


def _fake_qc(verdict="PASS", duration_s=5.0):
    return {"duration_s": duration_s, "sample_rate": 24000, "clip_frac": 0.0,
            "rms_dbfs": -20.0, "snr_db": 25.0, "verdict": verdict, "problems": []}


@pytest.fixture(autouse=True)
def _stub_audio_io(monkeypatch):
    """Every intake test in this file works with placeholder (non-audio)
    files -- ffmpeg-backed decode/QC is replaced with deterministic stubs."""
    monkeypatch.setattr(voice_dataset, "_read_wav_mono", lambda path: (np.zeros(10), 24000))
    monkeypatch.setattr(voice_dataset, "_qc_clip", lambda audio, rate, noise_floor_db: _fake_qc())


# ----------------------------------------------------
# _load_extra_transcripts
# ----------------------------------------------------

def test_load_extra_transcripts_returns_empty_for_none():
    assert voice_dataset._load_extra_transcripts(None) == {}


def test_load_extra_transcripts_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        voice_dataset._load_extra_transcripts(str(tmp_path / "nope.json"))


def test_load_extra_transcripts_accepts_plain_string_values(tmp_path):
    p = tmp_path / "transcripts.json"
    p.write_text(json.dumps({"EXTRA_1": "Some freeform line."}), encoding="utf-8")
    result = voice_dataset._load_extra_transcripts(str(p))
    assert result == {"EXTRA_1": ("Neutral", "Some freeform line.")}


def test_load_extra_transcripts_accepts_dict_values_with_emotion(tmp_path):
    p = tmp_path / "transcripts.json"
    p.write_text(json.dumps({"EXTRA_1": {"emotion": "Joy", "transcript": "Hooray!"}}), encoding="utf-8")
    result = voice_dataset._load_extra_transcripts(str(p))
    assert result == {"EXTRA_1": ("Joy", "Hooray!")}


# ----------------------------------------------------
# cmd_intake corpus accumulation
# ----------------------------------------------------

def test_intake_rejects_transcripts_colliding_with_fixed_prompt_ids(dataset_root):
    name = "Actor1"
    _init_dataset(name)
    input_dir = dataset_root / "session1"
    input_dir.mkdir()

    transcripts_path = dataset_root / "transcripts.json"
    transcripts_path.write_text(json.dumps({"A01": "Collides with the fixed script."}), encoding="utf-8")

    with pytest.raises(SystemExit):
        voice_dataset.cmd_intake(name, str(input_dir), str(transcripts_path))


def test_intake_accepts_extra_freeform_clip_via_transcripts(dataset_root):
    name = "Actor1"
    _init_dataset(name)
    input_dir = dataset_root / "session1"
    input_dir.mkdir()
    (input_dir / "EXT_001.wav").write_bytes(b"fake-audio")

    transcripts_path = dataset_root / "transcripts.json"
    transcripts_path.write_text(json.dumps({"EXT_001": "An extra freeform sentence."}), encoding="utf-8")

    voice_dataset.cmd_intake(name, str(input_dir), str(transcripts_path))

    qc_path = os.path.join(voice_dataset._dataset_dir(name), "qc_report.json")
    with open(qc_path, encoding="utf-8") as f:
        report = json.load(f)["clips"]
    assert "EXT_001" in report
    assert report["EXT_001"]["transcript"] == "An extra freeform sentence."
    assert os.path.exists(os.path.join(voice_dataset._dataset_dir(name), "raw", "EXT_001.wav"))


def test_intake_accumulates_across_multiple_sessions(dataset_root):
    """Two separate intake calls (e.g. the base script, then an extra
    freeform session) must both contribute to the same qc_report.json rather
    than the second overwriting the first -- this is what lets a corpus grow
    toward the 30-60 minute fine-tuning threshold."""
    name = "Actor1"
    _init_dataset(name)

    session1 = dataset_root / "session1"
    session1.mkdir()
    (session1 / "A01.wav").write_bytes(b"fake-audio")
    voice_dataset.cmd_intake(name, str(session1))

    session2 = dataset_root / "session2"
    session2.mkdir()
    (session2 / "EXT_001.wav").write_bytes(b"fake-audio")
    transcripts_path = dataset_root / "transcripts.json"
    transcripts_path.write_text(json.dumps({"EXT_001": "Extra sentence from session 2."}), encoding="utf-8")
    voice_dataset.cmd_intake(name, str(session2), str(transcripts_path))

    qc_path = os.path.join(voice_dataset._dataset_dir(name), "qc_report.json")
    with open(qc_path, encoding="utf-8") as f:
        report = json.load(f)["clips"]
    assert "A01" in report
    assert "EXT_001" in report


# ----------------------------------------------------
# CLI wrappers for the fine-tune job lifecycle
# ----------------------------------------------------

def test_cmd_finetune_delegates_to_voice_finetune(monkeypatch, capsys):
    calls = {}

    def fake_start(name, epochs, batch_size):
        calls["args"] = (name, epochs, batch_size)
        return {"state": "queued", "name": name}

    import src.voice_finetune as vf
    monkeypatch.setattr(vf, "start_finetune_job", fake_start)
    monkeypatch.setattr(vf, "DEFAULT_EPOCHS", 6)
    monkeypatch.setattr(vf, "DEFAULT_BATCH_SIZE", 4)

    voice_dataset.cmd_finetune("Actor1", epochs=None, batch_size=None)
    assert calls["args"] == ("Actor1", 6, 4)
    assert "queued" in capsys.readouterr().out


def test_cmd_finetune_reports_value_error_cleanly(monkeypatch):
    import src.voice_finetune as vf

    def fake_start(*args, **kwargs):
        raise ValueError("not enough audio")

    monkeypatch.setattr(vf, "start_finetune_job", fake_start)
    with pytest.raises(SystemExit, match="not enough audio"):
        voice_dataset.cmd_finetune("Actor1", epochs=None, batch_size=None)


def test_cmd_finetune_status_prints_status(monkeypatch, capsys):
    import src.voice_finetune as vf
    monkeypatch.setattr(vf, "get_finetune_status", lambda name: {"state": "ready", "name": name})
    voice_dataset.cmd_finetune_status("Actor1")
    out = capsys.readouterr().out
    assert "ready" in out


def test_cmd_cancel_finetune_reports_result(monkeypatch, capsys):
    import src.voice_finetune as vf
    monkeypatch.setattr(vf, "cancel_finetune_job", lambda name: True)
    voice_dataset.cmd_cancel_finetune("Actor1")
    assert "Cancelled" in capsys.readouterr().out

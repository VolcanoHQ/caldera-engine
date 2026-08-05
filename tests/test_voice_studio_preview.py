#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import types

import pytest

sys.modules.setdefault(
    "src.voice_dataset",
    types.SimpleNamespace(
        DATASET_ROOT="scratch/test_voice_datasets",
        _prompts=lambda: [],
        _qc_clip=lambda *_args, **_kwargs: {},
        _read_wav_mono=lambda *_args, **_kwargs: ([], 48000),
        cmd_build=lambda *_args, **_kwargs: None,
        cmd_init=lambda *_args, **_kwargs: None,
    ),
)

from src import voice_studio


def _seed_session(tmp_path, name="demo"):
    session = name
    voice_studio.DATASET_ROOT = str(tmp_path / "voice_datasets")
    session_dir = tmp_path / "voice_datasets" / session
    refs_dir = session_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "reference_mono.wav").write_bytes(b"fake")
    state = {
        "session": session,
        "speaker": session,
        "owner": "local",
        "clips": {},
        "questionnaire": {},
        "personas": [],
        "sfx": {},
        "built": True,
        "published": False,
    }
    with open(session_dir / "studio_session.json", "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    return session


def test_preview_rejects_mock_tone(monkeypatch, tmp_path):
    session = _seed_session(tmp_path)

    class _FakePalace:
        def get_character_drawer(self, *_args, **_kwargs):
            return {"voice_ref_path": "ok.wav"}

        def register_character(self, *_args, **_kwargs):
            return True

    class _FakeSynth:
        def __init__(self):
            self.palace = _FakePalace()

        def synthesize_line(self, **kwargs):
            with open(kwargs["output_wav_path"], "wb") as handle:
                handle.write(b"fakewav")
            return {"engine": "mock_tone"}

    monkeypatch.setitem(sys.modules, "src.voice_synthesizer", types.SimpleNamespace(VoiceSynthesizer=_FakeSynth))

    with pytest.raises(ValueError, match="synthetic tone"):
        voice_studio.preview(session, "hello world")


def test_preview_returns_engine_when_speech_generated(monkeypatch, tmp_path):
    session = _seed_session(tmp_path, name="demo2")

    class _FakePalace:
        def get_character_drawer(self, *_args, **_kwargs):
            return {"voice_ref_path": "ok.wav"}

        def register_character(self, *_args, **_kwargs):
            return True

    class _FakeSynth:
        def __init__(self):
            self.palace = _FakePalace()

        def synthesize_line(self, **kwargs):
            with open(kwargs["output_wav_path"], "wb") as handle:
                handle.write(b"fakewav")
            return {"engine": "edge_tts"}

    monkeypatch.setitem(sys.modules, "src.voice_synthesizer", types.SimpleNamespace(VoiceSynthesizer=_FakeSynth))

    result = voice_studio.preview(session, "hello world")
    assert result["engine"] == "edge_tts"
    assert os.path.exists(result["wav"])

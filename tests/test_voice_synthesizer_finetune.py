"""Tests for src/voice_synthesizer.py's fine-tuned-checkpoint-aware model
selection (_get_synthesis_model): picks a per-character fine-tuned XTTS
checkpoint (src/voice_finetune.py's premium tier) when the character's
MemPalace drawer has one pinned and it loads successfully, and always falls
back safely to the shared zero-shot model otherwise. Never exercises a real
GPU/model load -- torch/TTS are faked via sys.modules injection so this stays
hermetic in an environment with no ML deps installed.
"""

import os
import sys
import types

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import voice_synthesizer


@pytest.fixture
def synth(tmp_path):
    return voice_synthesizer.VoiceSynthesizer(mempalace_path=str(tmp_path / "mempalace"), force_cpu=True)


def test_no_checkpoint_dir_uses_shared_model(synth):
    shared_model = object()
    synth.xtts_model = shared_model
    result = synth._get_synthesis_model({"finetuned_checkpoint_dir": None})
    assert result is shared_model


def test_missing_char_drawer_uses_shared_model(synth):
    shared_model = object()
    synth.xtts_model = shared_model
    result = synth._get_synthesis_model({})
    assert result is shared_model


def test_checkpoint_dir_missing_on_disk_falls_back_to_shared_model(synth, tmp_path):
    shared_model = object()
    synth.xtts_model = shared_model
    nonexistent = str(tmp_path / "no_such_checkpoint")
    result = synth._get_synthesis_model({"finetuned_checkpoint_dir": nonexistent})
    assert result is shared_model


def test_checkpoint_dir_present_but_no_torch_falls_back_to_shared_model(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(voice_synthesizer, "HAS_TORCH", False)
    shared_model = object()
    synth.xtts_model = shared_model
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    result = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(checkpoint_dir)})
    assert result is shared_model


def _install_fake_tts_module(monkeypatch, loaded_models):
    """Injects a fake TTS.api module so `from TTS.api import TTS` succeeds
    without the real (uninstalled) coqui-tts package."""

    class _FakeTTS:
        def __init__(self, model_path=None, config_path=None):
            self.model_path = model_path
            self.config_path = config_path
            loaded_models.append(self)

        def to(self, device):
            self.device = device
            return self

    fake_tts_api = types.SimpleNamespace(TTS=_FakeTTS)
    fake_tts_pkg = types.ModuleType("TTS")
    fake_tts_pkg.api = fake_tts_api
    monkeypatch.setitem(sys.modules, "TTS", fake_tts_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_tts_api)
    return _FakeTTS


def test_checkpoint_dir_present_with_torch_loads_and_caches(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(voice_synthesizer, "HAS_TORCH", True)
    loaded_models = []
    _install_fake_tts_module(monkeypatch, loaded_models)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    shared_model = object()
    synth.xtts_model = shared_model

    result1 = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(checkpoint_dir)})
    assert result1 is not shared_model
    assert len(loaded_models) == 1  # actually loaded once

    # Second call for the same checkpoint should hit the cache, not reload.
    result2 = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(checkpoint_dir)})
    assert result2 is result1
    assert len(loaded_models) == 1


def test_checkpoint_load_failure_falls_back_to_shared_model(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(voice_synthesizer, "HAS_TORCH", True)

    class _ExplodingTTS:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("corrupt checkpoint")

    fake_tts_api = types.SimpleNamespace(TTS=_ExplodingTTS)
    fake_tts_pkg = types.ModuleType("TTS")
    fake_tts_pkg.api = fake_tts_api
    monkeypatch.setitem(sys.modules, "TTS", fake_tts_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_tts_api)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    shared_model = object()
    synth.xtts_model = shared_model
    result = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(checkpoint_dir)})
    assert result is shared_model


def test_model_cache_evicts_lru_beyond_cache_size(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(voice_synthesizer, "HAS_TORCH", True)
    loaded_models = []
    _install_fake_tts_module(monkeypatch, loaded_models)
    synth._finetuned_model_cache_size = 1

    dir_a = tmp_path / "checkpoint_a"
    dir_b = tmp_path / "checkpoint_b"
    dir_a.mkdir()
    dir_b.mkdir()

    model_a = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(dir_a)})
    model_b = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(dir_b)})
    assert model_a is not model_b
    assert len(synth._finetuned_model_cache) == 1
    assert str(dir_a) not in synth._finetuned_model_cache
    assert str(dir_b) in synth._finetuned_model_cache

    # Reloading dir_a must load a fresh instance (it was evicted), not reuse model_a.
    model_a_again = synth._get_synthesis_model({"finetuned_checkpoint_dir": str(dir_a)})
    assert model_a_again is not model_a

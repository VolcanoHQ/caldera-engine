"""Tests for MemPalace's finetuned_checkpoint_dir column/getter/setter
(src/spatial_memory.py) -- backs the per-voice XTTS fine-tuning premium tier
(src/voice_finetune.py) so voice_synthesizer.py knows which characters have a
real trained checkpoint vs. the default zero-shot cloning path.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.spatial_memory import MemPalace


@pytest.fixture
def palace(tmp_path):
    # use_chroma=False keeps this test fast/hermetic (no vector index needed
    # for a plain relational column round-trip) and avoids requiring chromadb.
    return MemPalace(db_dir=str(tmp_path / "mempalace"), use_chroma=False)


def test_get_character_drawer_returns_none_finetuned_checkpoint_by_default(palace):
    palace.register_character("Aria", voice_ref_path="does_not_exist.wav")
    drawer = palace.get_character_drawer("Aria")
    assert drawer is not None
    assert drawer["finetuned_checkpoint_dir"] is None


def test_set_finetuned_checkpoint_round_trips(palace):
    palace.register_character("Aria", voice_ref_path="does_not_exist.wav")
    ok = palace.set_finetuned_checkpoint("Aria", "data/voice_datasets/Aria/finetune/checkpoint")
    assert ok is True

    drawer = palace.get_character_drawer("Aria")
    assert drawer["finetuned_checkpoint_dir"] == "data/voice_datasets/Aria/finetune/checkpoint"


def test_set_finetuned_checkpoint_can_clear_with_none(palace):
    palace.register_character("Aria", voice_ref_path="does_not_exist.wav")
    palace.set_finetuned_checkpoint("Aria", "some/checkpoint/dir")
    palace.set_finetuned_checkpoint("Aria", None)

    drawer = palace.get_character_drawer("Aria")
    assert drawer["finetuned_checkpoint_dir"] is None


def test_set_finetuned_checkpoint_returns_false_for_unregistered_character(palace):
    ok = palace.set_finetuned_checkpoint("NoSuchCharacter", "some/checkpoint/dir")
    assert ok is False


def test_finetuned_checkpoint_does_not_affect_other_drawer_fields(palace):
    palace.register_character("Aria", voice_ref_path="does_not_exist.wav", speed=1.2, pitch=0.5)
    palace.set_finetuned_checkpoint("Aria", "some/checkpoint/dir")

    drawer = palace.get_character_drawer("Aria")
    assert drawer["voice_ref_path"] == "does_not_exist.wav"
    assert drawer["modulation_config"]["speed"] == 1.2
    assert drawer["modulation_config"]["pitch"] == 0.5

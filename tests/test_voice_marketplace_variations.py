#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Voice Marketplace variation model (recorded + algorithmic).

qdrant_client and ffmpeg are not available in this dev sandbox, so these tests
construct a VoiceMarketplace instance without calling __init__ (which opens a
real Qdrant client) and monkeypatch the Qdrant/ffmpeg boundary methods
(get_listing, generate_clap_embedding, self.client.upsert, _denoise_and_concat,
_render_algorithmic_variant) so the *logic under test* -- validation rules,
payload shape, id derivation -- runs for real.
"""

import pytest

from src.voice_marketplace import VoiceMarketplace


def _bare_marketplace():
    """A VoiceMarketplace with no real Qdrant/filesystem/ffmpeg dependency."""
    m = VoiceMarketplace.__new__(VoiceMarketplace)
    m.client = _FakeQdrantClient()
    return m


class _FakeQdrantClient:
    def __init__(self):
        self.upserted = []

    def upsert(self, collection_name, points):
        self.upserted.extend(points)


BASE_LISTING = {
    "voice_id": "base-1",
    "voice_name": "Old Salt Captain",
    "voice_ref_path": "data/voice_references/captain_mono.wav",
    "description": "A raspy elderly British sea captain voice",
    "seller": "Jane Seller",
    "seller_id": "seller-42",
    "price_usd": 25.0,
    "consent_confirmed": True,
    "parent_voice_id": None,
}


def test_onboard_variation_rejects_missing_parent(monkeypatch):
    m = _bare_marketplace()
    monkeypatch.setattr(m, "get_listing", lambda voice_id: None)

    with pytest.raises(ValueError, match="No such base voice listing"):
        m.onboard_variation("missing-parent", "excited", sample_wav_paths=["a.wav"])


def test_onboard_variation_rejects_variation_of_a_variation(monkeypatch):
    m = _bare_marketplace()
    child_listing = dict(BASE_LISTING, parent_voice_id="base-1")
    monkeypatch.setattr(m, "get_listing", lambda voice_id: child_listing)

    with pytest.raises(ValueError, match="not another variation"):
        m.onboard_variation("some-variation-id", "extra_excited", sample_wav_paths=["a.wav"])


@pytest.mark.parametrize("kwargs", [
    {},  # neither recorded nor algorithmic
    {"sample_wav_paths": ["a.wav"], "algorithmic_preset": "pitch_up"},  # both
])
def test_onboard_variation_requires_exactly_one_mode(monkeypatch, kwargs):
    m = _bare_marketplace()
    monkeypatch.setattr(m, "get_listing", lambda voice_id: dict(BASE_LISTING))

    with pytest.raises(ValueError, match="exactly one of"):
        m.onboard_variation("base-1", "some_label", **kwargs)


def test_onboard_variation_rejects_unknown_preset(monkeypatch):
    m = _bare_marketplace()
    monkeypatch.setattr(m, "get_listing", lambda voice_id: dict(BASE_LISTING))

    with pytest.raises(ValueError, match="Unknown algorithmic_preset"):
        m.onboard_variation("base-1", "weird", algorithmic_preset="not_a_real_preset")


def test_onboard_variation_recorded_writes_expected_payload(monkeypatch):
    m = _bare_marketplace()
    monkeypatch.setattr(m, "get_listing", lambda voice_id: dict(BASE_LISTING))
    monkeypatch.setattr(m, "generate_clap_embedding", lambda text: [0.0] * 128)
    monkeypatch.setattr(m, "_denoise_and_concat", lambda paths, out_dir, out_name="reference_mono.wav": (
        f"{out_dir}/reference_mono.wav", 12.5
    ))

    result = m.onboard_variation(
        "base-1", "excited_take",
        description="An excited, energetic delivery",
        price_usd=9.99,
        sample_wav_paths=["sample1.wav", "sample2.wav"],
    )

    assert result["variation_type"] == "recorded"
    assert result["parent_voice_id"] == "base-1"

    assert len(m.client.upserted) == 1
    point = m.client.upserted[0]
    assert point.payload["variation_type"] == "recorded"
    assert point.payload["variation_label"] == "excited_take"
    assert point.payload["parent_voice_id"] == "base-1"
    assert point.payload["sample_seconds"] == 12.5
    # Variations inherit seller/consent from the parent listing rather than
    # requiring the buyer to re-confirm consent per-variation.
    assert point.payload["seller"] == "Jane Seller"
    assert point.payload["consent_confirmed"] is True
    assert point.payload["price_usd"] == 9.99


def test_onboard_variation_algorithmic_preset_writes_expected_payload(monkeypatch):
    m = _bare_marketplace()
    monkeypatch.setattr(m, "get_listing", lambda voice_id: dict(BASE_LISTING))
    monkeypatch.setattr(m, "generate_clap_embedding", lambda text: [0.0] * 128)
    monkeypatch.setattr(m, "_render_algorithmic_variant", lambda source_wav, out_dir, params: f"{out_dir}/variant.wav")

    result = m.onboard_variation(
        "base-1", "pitch_up",
        algorithmic_preset="pitch_up",
        price_usd=4.99,
    )

    assert result["variation_type"] == "algorithmic"
    point = m.client.upserted[0]
    assert point.payload["variation_type"] == "algorithmic"
    assert point.payload["sample_seconds"] is None
    assert point.payload["price_usd"] == 4.99


def test_onboard_variation_algorithmic_explicit_params(monkeypatch):
    m = _bare_marketplace()
    monkeypatch.setattr(m, "get_listing", lambda voice_id: dict(BASE_LISTING))
    monkeypatch.setattr(m, "generate_clap_embedding", lambda text: [0.0] * 128)

    captured_params = {}

    def fake_render(source_wav, out_dir, params):
        captured_params.update(params)
        return f"{out_dir}/variant.wav"

    monkeypatch.setattr(m, "_render_algorithmic_variant", fake_render)

    m.onboard_variation("base-1", "custom_deep", pitch_semitones=-5.0, speed_ratio=0.8)

    assert captured_params == {"pitch_semitones": -5.0, "speed_ratio": 0.8}


def test_algorithmic_presets_have_pitch_and_speed_for_every_entry():
    for name, params in VoiceMarketplace.ALGORITHMIC_PRESETS.items():
        assert "pitch_semitones" in params, name
        assert "speed_ratio" in params, name
        assert params["speed_ratio"] > 0, name


def test_point_to_listing_surfaces_variation_fields():
    m = _bare_marketplace()

    class FakePoint:
        id = "var-1"
        payload = {
            "voice_name": "Old Salt Captain — excited_take",
            "voice_ref_path": "data/voice_marketplace/base-1/variations/var-1/reference_mono.wav",
            "description": "An excited take",
            "seller": "Jane Seller",
            "seller_id": "seller-42",
            "price_usd": 9.99,
            "consent_confirmed": True,
            "parent_voice_id": "base-1",
            "variation_type": "recorded",
            "variation_label": "excited_take",
            "sample_seconds": 12.5,
        }

    listing = m._point_to_listing(FakePoint())
    assert listing["parent_voice_id"] == "base-1"
    assert listing["variation_type"] == "recorded"
    assert listing["variation_label"] == "excited_take"


def test_point_to_listing_base_voice_has_no_parent():
    m = _bare_marketplace()

    class FakePoint:
        id = "base-1"
        payload = dict(BASE_LISTING, parent_voice_id=None)

    listing = m._point_to_listing(FakePoint())
    assert listing["parent_voice_id"] is None
    assert listing["variation_type"] is None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Voice Marketplace's real text-embedding backend (Phase 2 of the
marketplace build-out): dimension/normalization properties of
generate_clap_embedding, and the collection-dimension migration that runs when
EMBED_DIM changes (e.g. swapping the old keyword-simulation embedding for a
real model).

These tests use a real (embedded, on-disk) Qdrant instance under a pytest
tmp_path -- qdrant_client's local mode needs no server -- so the migration
logic is exercised against actual Qdrant behavior rather than a mock. torch/
transformers may or may not be installed in the environment running these
tests; assertions only rely on properties that hold for both the real model
and the deterministic hash-based fallback (fixed dimensionality, unit-norm,
determinism for a fixed input).
"""

import os

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.voice_marketplace import EMBED_DIM, VoiceMarketplace


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "qdrant_db")


def test_generate_clap_embedding_has_expected_dimension_and_is_unit_norm(db_path):
    m = VoiceMarketplace(db_path=db_path)
    try:
        vector = m.generate_clap_embedding("a warm elderly British narrator")
        assert len(vector) == EMBED_DIM
        norm = float(np.linalg.norm(vector))
        assert norm == pytest.approx(1.0, abs=1e-4)
    finally:
        m.client.close()


def test_generate_clap_embedding_is_deterministic(db_path):
    m = VoiceMarketplace(db_path=db_path)
    try:
        text = "a bright young energetic teen narrator"
        assert m.generate_clap_embedding(text) == m.generate_clap_embedding(text)
    finally:
        m.client.close()


def test_new_collection_is_created_at_embed_dim(db_path):
    m = VoiceMarketplace(db_path=db_path)
    try:
        info = m.client.get_collection("voice_marketplace")
        assert info.config.params.vectors.size == EMBED_DIM
    finally:
        m.client.close()


def test_migrates_existing_collection_at_stale_dimension(db_path):
    """
    Simulates a marketplace database created before the real-embedding upgrade
    (128-dim keyword-simulation vectors): the next VoiceMarketplace() startup
    should detect the mismatch, recreate the collection at EMBED_DIM, and
    re-embed every existing listing's description without losing its payload.
    """
    stale_dim = 128
    setup_client = QdrantClient(path=db_path)
    setup_client.create_collection(
        collection_name="voice_marketplace",
        vectors_config=VectorParams(size=stale_dim, distance=Distance.COSINE),
    )
    setup_client.upsert(
        collection_name="voice_marketplace",
        points=[
            PointStruct(
                id=1,
                vector=[0.1] * stale_dim,
                payload={
                    "voice_name": "Old Salt Captain",
                    "description": "a gruff old sailor with a raspy voice",
                },
            )
        ],
    )
    setup_client.close()

    m = VoiceMarketplace(db_path=db_path)
    try:
        info = m.client.get_collection("voice_marketplace")
        assert info.config.params.vectors.size == EMBED_DIM

        listing = m.get_listing(1)
        assert listing is not None
        assert listing["voice_name"] == "Old Salt Captain"
        assert listing["description"] == "a gruff old sailor with a raspy voice"

        # Migrated collection should still be usable for real (new-dimension) writes/reads.
        results = m.search_marketplace("a gruff old sailor with a raspy voice", limit=1)
        assert results
        assert results[0]["voice_name"] == "Old Salt Captain"

        new_id = m.register_voice("Fresh Voice", "refs/fresh.wav", "a bright young energetic teen")
        assert m.get_listing(new_id) is not None
    finally:
        m.client.close()


def test_no_migration_when_dimension_already_matches(db_path):
    """Startup should not touch a collection that's already at EMBED_DIM."""
    m1 = VoiceMarketplace(db_path=db_path)
    voice_id = m1.register_voice("Stable Voice", "refs/stable.wav", "a calm neutral narrator")
    m1.client.close()

    m2 = VoiceMarketplace(db_path=db_path)
    try:
        listing = m2.get_listing(voice_id)
        assert listing is not None
        assert listing["voice_name"] == "Stable Voice"
    finally:
        m2.client.close()

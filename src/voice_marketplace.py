#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Voice Marketplace

*** DEPRECATED as the live backing implementation. ***
The Voice Marketplace has been extracted into its own standalone product
("Volcano Studios Voice Marketplace" -- separate repo, runs via Docker).
gui_server.py and console_api.py now talk to that product over HTTP via
src/marketplace_client.py (MarketplaceClient), NOT this module. This file is
kept in place only because its existing tests
(tests/test_voice_marketplace_variations.py, test_voice_marketplace_embeddings.py,
test_voice_marketplace_stripe_integration.py) still exercise it directly and
document the original in-process design; it is not imported by any live
request-handling code path anymore. Do not add new features here -- make
changes in the standalone product's src/volcano_marketplace/marketplace.py
instead (the two started as a fork of this file and should be kept in sync
by hand for now).

Uses Qdrant vector database to store and semantically retrieve voice marketplace audio references
by projecting user description prompts into a text embedding space, so buyers can search with
natural language ("cozy elderly narrator for a children's book") and match listings that don't
share exact keywords with the query.
"""

import os
import re
import uuid
import shutil
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

if TYPE_CHECKING:
    from src.marketplace_payments import MarketplacePayments

logger = logging.getLogger("VoiceMarketplace")

# Dimensionality of the marketplace's semantic search vectors. Matches the hidden size of
# sentence-transformers/all-MiniLM-L6-v2, the real text-embedding model used when
# torch/transformers are available. If this model is ever swapped for another one with a
# different hidden size (or for a real audio-CLAP model embedding the WAV itself), bump this
# constant -- _migrate_if_dimension_mismatch() will automatically recreate the Qdrant collection
# and re-embed every existing listing the next time the marketplace starts up.
EMBED_DIM = 384
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Graceful optional dependency loading, matching the pattern used elsewhere in this codebase
# (see src/hybrid_nlp_pipeline.py): the marketplace works everywhere, but only does real
# semantic generalization when torch/transformers are installed.
HAS_TEXT_EMBEDDING_MODEL = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TEXT_EMBEDDING_MODEL = True
except ImportError:
    logger.warning(
        "torch/transformers not found in environment. Voice Marketplace semantic search will "
        "fall back to a deterministic hash-based pseudo-embedding (no real semantic "
        "generalization) until those packages are installed."
    )

_embed_tokenizer = None
_embed_model = None


def _load_embedding_model():
    """Lazily loads and caches the sentence-embedding model/tokenizer (process-wide singleton)."""
    global _embed_tokenizer, _embed_model
    if _embed_model is None:
        _embed_tokenizer = AutoTokenizer.from_pretrained(_EMBED_MODEL_NAME)
        _embed_model = AutoModel.from_pretrained(_EMBED_MODEL_NAME)
        _embed_model.eval()
    return _embed_tokenizer, _embed_model


def _mean_pool(token_embeddings, attention_mask):
    """Mean-pools token embeddings weighted by the attention mask (standard sentence-transformers pooling)."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class VoiceMarketplace:
    """Manages the Voice Marketplace vector database using Qdrant."""

    def __init__(self, db_path: str = "data/qdrant_db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize Qdrant Client (running locally in storage directory)
        self.client = QdrantClient(path=self.db_path)
        self._init_collection()
        self._payments = None  # lazily constructed -- see .payments property

    @property
    def payments(self) -> "MarketplacePayments":
        """Lazily-constructed Stripe Connect payments client. Lazy so importing
        VoiceMarketplace (and its many callers/tests that never touch payments)
        doesn't require the `stripe` package or a configured .env just to run
        search/casting/variations."""
        if self._payments is None:
            from src.marketplace_payments import MarketplacePayments
            self._payments = MarketplacePayments()
        return self._payments

    def _init_collection(self):
        """Creates the voice_marketplace collection if it doesn't already exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if "voice_marketplace" not in collection_names:
                logger.info("Creating 'voice_marketplace' collection in Qdrant...")
                self.client.create_collection(
                    collection_name="voice_marketplace",
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
                )
                self._seed_default_marketplace()
            else:
                self._migrate_if_dimension_mismatch()
                logger.info("Qdrant collection 'voice_marketplace' already exists.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    def _migrate_if_dimension_mismatch(self):
        """
        One-time migration guard for when EMBED_DIM changes (e.g. swapping the old keyword
        simulation for a real embedding model, or later upgrading to audio-CLAP): backs up
        every listing's payload, recreates the collection at the new dimensionality, and
        re-embeds every listing's description with the current embedding backend.
        """
        try:
            info = self.client.get_collection("voice_marketplace")
            current_size = info.config.params.vectors.size
        except Exception as e:
            logger.warning(f"Could not inspect existing collection vector size ({e}); skipping migration check.")
            return

        if current_size == EMBED_DIM:
            return

        logger.warning(
            f"Voice Marketplace embedding dimension changed ({current_size} -> {EMBED_DIM}); "
            "migrating collection and re-embedding all listings..."
        )
        points, _ = self.client.scroll(
            collection_name="voice_marketplace", limit=10000, with_payload=True, with_vectors=False
        )

        # Local-mode (embedded) Qdrant on Windows can leave storage.sqlite locked by the
        # already-loaded collection object, so delete_collection()'s internal rmtree silently
        # no-ops (ignore_errors=True) and a same-instance create_collection() at the new
        # dimensionality ends up reusing the stale on-disk vectors array, corrupting the
        # upsert below. Fully closing the client first releases that lock, so the manual
        # rmtree actually removes the files; reopening a fresh client and calling
        # delete_collection() again (a no-op on the already-removed directory) also clears
        # the client's in-memory/meta.json bookkeeping before we recreate the collection.
        self.client.close()
        collection_dir = os.path.join(self.db_path, "collection", "voice_marketplace")
        shutil.rmtree(collection_dir, ignore_errors=True)
        self.client = QdrantClient(path=self.db_path)
        self.client.delete_collection("voice_marketplace")
        self.client.create_collection(
            collection_name="voice_marketplace",
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
        if points:
            new_points = []
            for p in points:
                payload = p.payload or {}
                vector = self.generate_clap_embedding(payload.get("description", ""))
                new_points.append(PointStruct(id=p.id, vector=vector, payload=payload))
            self.client.upsert(collection_name="voice_marketplace", points=new_points)
            logger.info(f"Migrated {len(new_points)} marketplace listing(s) to the new embedding space.")

    def generate_clap_embedding(self, text: str) -> List[float]:
        """
        Real learned text embedding (sentence-transformers/all-MiniLM-L6-v2, mean-pooled,
        L2-normalized) over the voice's description. This generalizes to descriptions that
        don't share exact keywords -- e.g. "cozy elderly woman narrating a children's book"
        now scores close to "warm grandmother telling bedtime stories" -- unlike the old
        keyword-lookup simulation this replaces.

        Falls back to a deterministic hash-based pseudo-embedding when torch/transformers
        aren't installed, so the marketplace still functions (without real semantic
        generalization) in minimal environments.

        NOTE: this embeds the *text description*, not the audio itself. A future upgrade to a
        real audio-CLAP model (embedding the reference WAV's acoustic content directly) can
        replace the body of this method without touching any caller -- see EMBED_DIM above and
        _migrate_if_dimension_mismatch(), which handles re-embedding existing listings whenever
        the vector dimensionality changes.
        """
        if HAS_TEXT_EMBEDDING_MODEL:
            try:
                return self._model_text_embedding(text)
            except Exception as e:
                logger.warning(f"Text embedding model failed ({e}); falling back to hash-based pseudo-embedding for this call.")
        return self._fallback_hash_embedding(text)

    def _model_text_embedding(self, text: str) -> List[float]:
        """Real sentence embedding via a lazily-loaded, process-cached transformers model."""
        tokenizer, model = _load_embedding_model()
        inputs = tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        vector = torch.nn.functional.normalize(pooled, p=2, dim=1)[0]
        return vector.tolist()

    def _fallback_hash_embedding(self, text: str) -> List[float]:
        """Deterministic hash-based pseudo-embedding used only when no real model is available."""
        vector = np.zeros(EMBED_DIM, dtype=np.float32)
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        for idx in range(EMBED_DIM):
            vector[idx] = float(h[idx % len(h)]) / 255.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def register_voice(self, voice_name: str, voice_ref_path: str, description: str) -> str:
        """Saves a cloned voice in Qdrant with its CLAP embedding and metadata."""
        voice_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, voice_name))
        vector = self.generate_clap_embedding(description)
        
        point = PointStruct(
            id=voice_id,
            vector=vector,
            payload={
                "voice_name": voice_name,
                "voice_ref_path": voice_ref_path,
                "description": description
            }
        )
        
        self.client.upsert(
            collection_name="voice_marketplace",
            points=[point]
        )
        logger.info(f"Registered voice '{voice_name}' in Qdrant marketplace.")
        return voice_id

    def list_all(self, limit: int = 200, include_variations: bool = True) -> List[Dict[str, Any]]:
        """Every listing in the marketplace (browse view, no query). Base voices and
        their variations are both points in the same collection, distinguished by
        parent_voice_id (unset for a base voice). Catalog sizes here are small
        (in-process Qdrant), so the include_variations filter is applied in Python
        rather than via a fragile "key absent" Qdrant filter."""
        points, _ = self.client.scroll(
            collection_name="voice_marketplace",
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        listings = [self._point_to_listing(p) for p in points]
        if not include_variations:
            listings = [l for l in listings if not l.get("parent_voice_id")]
        return listings

    def _point_to_listing(self, p) -> Dict[str, Any]:
        payload = p.payload or {}
        return {
            "voice_id": p.id,
            "voice_name": payload.get("voice_name"),
            "voice_ref_path": payload.get("voice_ref_path"),
            "description": payload.get("description"),
            "seller": payload.get("seller"),
            "seller_id": payload.get("seller_id"),
            "price_usd": payload.get("price_usd"),
            "consent_confirmed": payload.get("consent_confirmed"),
            "parent_voice_id": payload.get("parent_voice_id") or None,
            "variation_type": payload.get("variation_type"),  # "recorded" | "algorithmic" | None (base voice)
            "variation_label": payload.get("variation_label"),
            "sample_seconds": payload.get("sample_seconds"),
        }

    def search_marketplace(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Queries Qdrant for semantic similarity match against acoustic-text embeddings."""
        query_vector = self.generate_clap_embedding(query)
        
        results = self.client.query_points(
            collection_name="voice_marketplace",
            query=query_vector,
            limit=limit
        )
        
        hits = []
        for hit in results.points:
            listing = self._point_to_listing(hit)
            listing["score"] = hit.score
            hits.append(listing)
            
        return hits

    def get_listing(self, voice_id: str) -> Optional[Dict[str, Any]]:
        points = self.client.retrieve(collection_name="voice_marketplace", ids=[voice_id])
        if not points:
            return None
        return self._point_to_listing(points[0])

    def list_variations(self, parent_voice_id: str) -> List[Dict[str, Any]]:
        """All variations (recorded + algorithmic) registered under a base voice,
        for the "select a delivery" step of casting or the browse detail view."""
        points, _ = self.client.scroll(
            collection_name="voice_marketplace",
            scroll_filter=Filter(must=[FieldCondition(key="parent_voice_id", match=MatchValue(value=parent_voice_id))]),
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_listing(p) for p in points]

    # ----------------------------------------------------
    # Seller onboarding: raw voice acting -> validated, licensed listing
    # ----------------------------------------------------

    def _denoise_and_concat(self, sample_wav_paths: List[str], out_dir: str, out_name: str = "reference_mono.wav") -> Tuple[str, float]:
        """Shared onboarding step: validate durations, denoise each sample, concatenate
        into one mono cloning reference. Used by both base-voice onboarding and
        recorded-variation onboarding. Returns (reference_wav_path, total_seconds)."""
        import wave as _wave
        import shutil
        import subprocess
        from src.voice_synthesizer import denoise_audio_file

        if not sample_wav_paths:
            raise ValueError("At least one sample WAV is required.")

        total_dur = 0.0
        for p in sample_wav_paths:
            if not os.path.exists(p):
                raise ValueError(f"Sample not found: {p}")
            with _wave.open(p) as w:
                total_dur += w.getnframes() / float(w.getframerate())
        if total_dur < 6.0:
            raise ValueError(f"Samples total {total_dur:.1f}s; at least 6s of clean speech is required for reliable cloning (30-60s recommended).")

        os.makedirs(out_dir, exist_ok=True)
        cleaned = []
        for i, p in enumerate(sample_wav_paths):
            out = os.path.join(out_dir, f"sample_{i}_clean.wav")
            try:
                if not denoise_audio_file(p, out):
                    shutil.copy(p, out)
            except Exception:
                shutil.copy(p, out)
            cleaned.append(out)
        concat_list = os.path.join(out_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in cleaned:
                f.write(f"file '{os.path.abspath(p)}'\n")
        reference_wav = os.path.join(out_dir, out_name)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-ac", "1", "-ar", "24000", reference_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return reference_wav, total_dur

    def onboard_voice(self, seller_name: str, voice_name: str, sample_wav_paths: List[str],
                      description: str, price_usd: float = 0.0, consent_confirmed: bool = False,
                      seller_id: str = "local") -> Dict[str, Any]:
        """Full seller pipeline: validate samples -> denoise -> assemble the cloning
        reference -> register a listing with provenance and consent metadata.

        Returns the listing record. Raises ValueError on unusable input -- a bad
        reference silently degrading every book it's cast in is the worst outcome.
        """
        if not consent_confirmed:
            raise ValueError("Seller consent must be explicitly confirmed before listing a voice.")

        voice_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seller_name}:{voice_name}"))
        voice_dir = os.path.join("data", "voice_marketplace", voice_id)
        reference_wav, total_dur = self._denoise_and_concat(sample_wav_paths, voice_dir)

        # Single listing under the seller-scoped id (register_voice would create a
        # second point under a name-only id -- duplicate listings)
        vector = self.generate_clap_embedding(description)
        self.client.upsert(collection_name="voice_marketplace", points=[PointStruct(
            id=voice_id, vector=vector, payload={
                "voice_name": voice_name,
                "voice_ref_path": reference_wav,
                "description": description,
                "seller": seller_name,
                "seller_id": seller_id or "local",
                "price_usd": price_usd,
                "consent_confirmed": True,
                "sample_seconds": round(total_dur, 1),
                "status": "listed",
            },
        )])
        logger.info(f"Onboarded voice '{voice_name}' by {seller_name} ({total_dur:.0f}s of samples) -> {reference_wav}")
        return {"voice_id": voice_id, "reference_wav": reference_wav, "sample_seconds": total_dur}

    # ----------------------------------------------------
    # Variations: real recorded emotional/delivery takes, and algorithmic
    # pitch/speed variants derived from the base (or another variation's) take.
    # ----------------------------------------------------

    ALGORITHMIC_PRESETS = {
        "pitch_up": {"pitch_semitones": 2.5, "speed_ratio": 1.0},
        "pitch_down": {"pitch_semitones": -2.5, "speed_ratio": 1.0},
        "faster": {"pitch_semitones": 0.0, "speed_ratio": 1.15},
        "slower": {"pitch_semitones": 0.0, "speed_ratio": 0.85},
        "deep_slow": {"pitch_semitones": -3.5, "speed_ratio": 0.92},
        "bright_fast": {"pitch_semitones": 3.5, "speed_ratio": 1.08},
    }

    def onboard_variation(
        self,
        parent_voice_id: str,
        variation_label: str,
        description: str = "",
        price_usd: float = 0.0,
        sample_wav_paths: Optional[List[str]] = None,
        algorithmic_preset: Optional[str] = None,
        pitch_semitones: Optional[float] = None,
        speed_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Registers a variation under an existing listing. Two mutually exclusive
        modes:
          - Recorded: pass sample_wav_paths (a real emotional/delivery take by the
            same seller) -- denoised and assembled exactly like a base voice.
          - Algorithmic: pass algorithmic_preset (one of ALGORITHMIC_PRESETS) or
            explicit pitch_semitones/speed_ratio -- derives the variant from the
            parent's reference WAV via the same ffmpeg pitch/speed filter chain
            voice_synthesizer.py uses at synthesis time, so previews match
            production output.
        """
        parent = self.get_listing(parent_voice_id)
        if not parent:
            raise ValueError(f"No such base voice listing: {parent_voice_id}")
        if parent.get("parent_voice_id"):
            raise ValueError("Variations must be attached to a base voice, not another variation.")

        recorded = bool(sample_wav_paths)
        algorithmic = bool(algorithmic_preset or pitch_semitones or speed_ratio)
        if recorded == algorithmic:
            raise ValueError("Provide exactly one of sample_wav_paths (recorded) or algorithmic_preset/pitch_semitones/speed_ratio (algorithmic).")

        variation_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_voice_id}:{variation_label}"))
        variation_dir = os.path.join("data", "voice_marketplace", parent_voice_id, "variations", variation_id)

        if recorded:
            reference_wav, total_dur = self._denoise_and_concat(sample_wav_paths, variation_dir)
            variation_type = "recorded"
            sample_seconds = round(total_dur, 1)
        else:
            os.makedirs(variation_dir, exist_ok=True)
            if algorithmic_preset:
                if algorithmic_preset not in self.ALGORITHMIC_PRESETS:
                    raise ValueError(f"Unknown algorithmic_preset: {algorithmic_preset}. Options: {sorted(self.ALGORITHMIC_PRESETS)}")
                params = self.ALGORITHMIC_PRESETS[algorithmic_preset]
            else:
                params = {"pitch_semitones": pitch_semitones or 0.0, "speed_ratio": speed_ratio or 1.0}
            reference_wav = self._render_algorithmic_variant(parent["voice_ref_path"], variation_dir, params)
            variation_type = "algorithmic"
            sample_seconds = None

        vector = self.generate_clap_embedding(description or f"{parent['voice_name']} ({variation_label})")
        self.client.upsert(collection_name="voice_marketplace", points=[PointStruct(
            id=variation_id, vector=vector, payload={
                "voice_name": f"{parent['voice_name']} — {variation_label}",
                "voice_ref_path": reference_wav,
                "description": description or f"{variation_label} delivery of {parent['voice_name']}",
                "seller": parent.get("seller"),
                "seller_id": parent.get("seller_id"),
                "price_usd": price_usd,
                "consent_confirmed": parent.get("consent_confirmed", False),
                "parent_voice_id": parent_voice_id,
                "variation_type": variation_type,
                "variation_label": variation_label,
                "sample_seconds": sample_seconds,
                "status": "listed",
            },
        )])
        logger.info(f"Onboarded {variation_type} variation '{variation_label}' under {parent_voice_id} -> {reference_wav}")
        return {"voice_id": variation_id, "parent_voice_id": parent_voice_id, "reference_wav": reference_wav, "variation_type": variation_type}

    def _render_algorithmic_variant(self, source_wav: str, out_dir: str, params: Dict[str, float]) -> str:
        """Applies a pitch/speed ffmpeg filter chain to a base reference WAV,
        reusing voice_synthesizer's filter-building logic so the preview a buyer
        hears matches what synthesis will actually produce."""
        import subprocess
        from src.voice_synthesizer import _build_pitch_speed_filters

        if not os.path.exists(source_wav):
            raise ValueError(f"Base voice reference not found: {source_wav}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "variant.wav")

        pitch_semitones = params.get("pitch_semitones", 0.0)
        speed_ratio = params.get("speed_ratio", 1.0)
        # voice_synthesizer expresses pitch as a ratio modifier (2^(semitones/12)),
        # matching the asetrate scheme _build_pitch_speed_filters expects.
        pitch_modifier = 2.0 ** (pitch_semitones / 12.0)
        filter_chain = _build_pitch_speed_filters(pitch_modifier, speed_ratio)

        cmd = ["ffmpeg", "-y", "-i", source_wav]
        if filter_chain:
            cmd += ["-af", filter_chain]
        cmd += ["-ac", "1", "-ar", "24000", out_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path

    # ----------------------------------------------------
    # Purchase / licensing ledger + casting integration
    # ----------------------------------------------------

    LICENSE_LEDGER = os.path.join("data", "voice_marketplace", "licenses.json")

    def purchase_voice(self, voice_id: str, buyer: str, purpose: str, payment_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Records a license grant (append-only ledger). `payment_info`, when
        provided, carries the Stripe session/payment_intent ids and confirmed
        charge amount from a real Checkout payment (see create_purchase_checkout
        / confirm_stripe_webhook below); it is omitted for the zero-friction
        internal casting path (director_cast_character / cast_character), which
        still records WHO was granted WHAT voice for WHAT purpose even when no
        real money changes hands (e.g. a $0 seed listing, or licensing your own
        listed voice)."""
        import json as _json
        import time as _time
        points = self.client.retrieve(collection_name="voice_marketplace", ids=[voice_id])
        if not points:
            raise ValueError(f"No such voice listing: {voice_id}")
        payload = points[0].payload
        license_record = {
            "license_id": str(uuid.uuid4()),
            "voice_id": voice_id,
            "voice_name": payload.get("voice_name"),
            "voice_ref_path": payload.get("voice_ref_path"),
            "seller": payload.get("seller", "(seed)"),
            "buyer": buyer,
            "purpose": purpose,
            "price_usd": payload.get("price_usd", 0.0),
            "granted_at": _time.time(),
            "payment": payment_info,  # None for internal/free grants; Stripe details for paid purchases
        }
        os.makedirs(os.path.dirname(self.LICENSE_LEDGER), exist_ok=True)
        ledger = []
        if os.path.exists(self.LICENSE_LEDGER):
            with open(self.LICENSE_LEDGER) as f:
                ledger = _json.load(f)
        ledger.append(license_record)
        with open(self.LICENSE_LEDGER, "w") as f:
            _json.dump(ledger, f, indent=2)
        logger.info(f"License granted: '{payload.get('voice_name')}' -> {buyer} for {purpose!r}")
        return license_record

    # ----------------------------------------------------
    # Stripe Connect: seller payouts + buyer checkout
    # ----------------------------------------------------

    def onboard_seller_payments(self, seller_id: str, email: str) -> Dict[str, Any]:
        """Starts (or resumes) Stripe Express Connect onboarding for a seller.
        Returns a one-time hosted URL the seller should be redirected to;
        Stripe collects identity/bank details directly."""
        return self.payments.create_seller_connect_account(seller_id, email)

    def get_seller_payment_status(self, seller_id: str) -> Dict[str, Any]:
        """Refreshes and returns a seller's Stripe Connect account status
        (charges_enabled/payouts_enabled/details_submitted)."""
        return self.payments.refresh_seller_account_status(seller_id)

    def create_purchase_checkout(self, voice_id: str, buyer_email: Optional[str] = None, purpose: str = "") -> Dict[str, Any]:
        """Creates a Stripe Checkout Session for a priced listing. The buyer pays
        the full listing price; Stripe automatically splits the platform fee and
        transfers the rest to the seller's connected account. The license is
        granted only once the webhook confirms payment (see
        confirm_stripe_webhook) -- never on the client-side redirect alone."""
        listing = self.get_listing(voice_id)
        if not listing:
            raise ValueError(f"No such voice listing: {voice_id}")
        seller_id = listing.get("seller_id")
        if not seller_id:
            raise ValueError(f"Listing {voice_id} has no seller_id on file; cannot route payment.")
        price_usd = listing.get("price_usd") or 0.0
        return self.payments.create_checkout_session(
            voice_id=voice_id,
            voice_name=listing.get("voice_name", voice_id),
            price_usd=price_usd,
            seller_id=seller_id,
            buyer_email=buyer_email,
            purpose=purpose,
        )

    def confirm_stripe_webhook(self, payload: bytes, sig_header: str) -> Optional[Dict[str, Any]]:
        """Verifies a Stripe webhook event and, if it's a completed/paid
        Checkout session for this marketplace, grants the corresponding
        license with the Stripe payment details attached. Returns None (and
        grants nothing) for any other event type -- Stripe sends many event
        types to a single webhook endpoint and callers should 200 those away."""
        from src.marketplace_payments import MarketplacePayments

        event = self.payments.construct_webhook_event(payload, sig_header)
        metadata = MarketplacePayments.extract_completed_checkout_metadata(event)
        if not metadata:
            return None

        payment_info = {
            "stripe_session_id": metadata["stripe_session_id"],
            "stripe_payment_intent_id": metadata["stripe_payment_intent_id"],
            "amount_total_usd": metadata["amount_total_usd"],
        }
        return self.purchase_voice(
            voice_id=metadata["voice_id"],
            buyer=metadata["buyer_email"] or "unknown_buyer",
            purpose=metadata["purpose"],
            payment_info=payment_info,
        )

    def cast_character(self, character_name: str, character_description: str,
                       buyer: str = "local", purpose: str = "") -> Optional[Dict[str, Any]]:
        """The marketplace<->production bridge: search listings by a character
        description (era/age/accent/register -- typically derived from the book
        bible + the character's role), license the best match, and bind it to the
        character's MemPalace drawer so every synthesis of that character uses the
        purchased voice. Zero-shot XTTS conditioning makes the 'training' instant."""
        hits = self.search_marketplace(character_description, limit=3)
        hits = [h for h in hits if os.path.exists(h["voice_ref_path"])]
        if not hits:
            logger.warning(f"No castable marketplace voice found for '{character_name}' ({character_description[:50]!r})")
            return None
        best = hits[0]
        return self.cast_character_with_voice(
            character_name=character_name,
            voice_id=best["voice_id"],
            buyer=buyer,
            purpose=purpose or f"cast as {character_name}",
            preselected_voice=best,
        )

    def cast_character_with_voice(
        self,
        character_name: str,
        voice_id: str,
        buyer: str = "local",
        purpose: str = "",
        preselected_voice: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        best = preselected_voice or self.get_listing(voice_id)
        if not best:
            raise ValueError(f"No such voice listing: {voice_id}")
        if not os.path.exists(best.get("voice_ref_path") or ""):
            raise ValueError(f"Voice reference path missing for listing: {voice_id}")
        license_record = self.purchase_voice(best["voice_id"], buyer=buyer, purpose=purpose or f"cast as {character_name}")

        from src.spatial_memory import MemPalace
        palace = MemPalace(use_chroma=False)
        try:
            import json as _json
            if not palace.get_character_drawer(character_name):
                palace.register_character(character_name=character_name, voice_ref_path=best["voice_ref_path"], speed=1.0, pitch=0.0)
            else:
                drawer = palace.get_character_drawer(character_name)
                config = drawer["modulation_config"]
                config.pop("xtts_speaker", None)  # purchased reference beats builtin pin
                palace.conn.execute(
                    "UPDATE drawers SET voice_ref_path = ?, modulation_config_json = ? WHERE character_name = ?",
                    (best["voice_ref_path"], _json.dumps(config), character_name),
                )
                palace.conn.commit()
        finally:
            palace.close()
        score_note = f" (score {best['score']:.2f})" if best.get("score") is not None else ""
        logger.info(f"Cast '{character_name}' with marketplace voice '{best['voice_name']}'{score_note}")
        return {"character": character_name, "voice": best, "license": license_record}

    def _seed_default_marketplace(self):
        """Seeds the Voice Marketplace with standard acoustic profiles."""
        logger.info("Seeding default voice profiles in Qdrant marketplace...")
        default_voices = [
            (
                "Old Salt Captain",
                "data/voice_references/captain_mono.wav",
                "A raspy, elderly British sea captain voice with deep nautical gravel and stern tone"
            ),
            (
                "Sherlock Holmes",
                "data/voice_references/holmes_mono.wav",
                "An energetic, commanding, authoritative British male voice with high intelligence tone"
            ),
            (
                "Dr. John Watson",
                "data/voice_references/watson_mono.wav",
                "A calm, soothing, deep British male narrator voice with mature stability"
            ),
            (
                "Maternal Storyteller",
                "data/voice_references/narrator_mono.wav",
                "A soft, soothing, calm female storyteller voice with warm maternal caring tone"
            ),
            (
                "Dublin Scholar",
                "data/voice_references/irish_scholar.wav",
                "A young, energetic Irish male voice with academic accent"
            ),
        ]
        for name, ref, desc in default_voices:
            self.register_voice(name, ref, desc)


def main():
    import argparse
    import json as _json
    from src.marketplace_payments import StripeNotConfiguredError
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Caldera Engine Voice Marketplace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("onboard", help="Seller: list a voice from sample WAVs")
    p.add_argument("--seller", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--price", type=float, default=0.0)
    p.add_argument("--consent", action="store_true", help="Seller confirms consent to license this voice")

    p = sub.add_parser("search", help="Browse listings by description")
    p.add_argument("query")

    p = sub.add_parser("cast", help="License the best-matching voice and bind it to a character drawer")
    p.add_argument("--character", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--buyer", default="local")

    p = sub.add_parser("onboard-variation", help="Seller: add a recorded or algorithmic variation to an existing listing")
    p.add_argument("--parent-voice-id", required=True)
    p.add_argument("--label", required=True, help="e.g. 'excited', 'whisper', 'pitch_up'")
    p.add_argument("--description", default="")
    p.add_argument("--price", type=float, default=0.0)
    p.add_argument("--samples", nargs="+", help="Recorded variation: sample WAV paths")
    p.add_argument("--preset", choices=sorted(VoiceMarketplace.ALGORITHMIC_PRESETS), help="Algorithmic variation preset")
    p.add_argument("--pitch-semitones", type=float, help="Algorithmic variation: explicit pitch shift")
    p.add_argument("--speed-ratio", type=float, help="Algorithmic variation: explicit speed ratio")

    p = sub.add_parser("list-variations", help="List variations under a base voice listing")
    p.add_argument("voice_id")

    p = sub.add_parser("onboard-seller-payments", help="Seller: connect a Stripe Express account for payouts")
    p.add_argument("--seller-id", required=True)
    p.add_argument("--email", required=True)

    p = sub.add_parser("seller-payment-status", help="Refresh a seller's Stripe Connect account status")
    p.add_argument("--seller-id", required=True)

    p = sub.add_parser("create-checkout", help="Buyer: create a Stripe Checkout session for a priced listing")
    p.add_argument("--voice-id", required=True)
    p.add_argument("--buyer-email")
    p.add_argument("--purpose", default="")

    args = parser.parse_args()
    m = VoiceMarketplace()
    if args.cmd == "onboard":
        print(_json.dumps(m.onboard_voice(args.seller, args.name, args.samples, args.description, args.price, args.consent), indent=2))
    elif args.cmd == "search":
        for h in m.search_marketplace(args.query):
            exists = "" if os.path.exists(h["voice_ref_path"]) else "  [reference missing]"
            print(f"{h['score']:.2f}  {h['voice_name']:24s} {h['description'][:60]}{exists}")
    elif args.cmd == "cast":
        result = m.cast_character(args.character, args.description, buyer=args.buyer)
        print(_json.dumps(result, indent=2, default=str) if result else "No castable voice found.")
    elif args.cmd == "onboard-variation":
        result = m.onboard_variation(
            parent_voice_id=args.parent_voice_id,
            variation_label=args.label,
            description=args.description,
            price_usd=args.price,
            sample_wav_paths=args.samples,
            algorithmic_preset=args.preset,
            pitch_semitones=args.pitch_semitones,
            speed_ratio=args.speed_ratio,
        )
        print(_json.dumps(result, indent=2))
    elif args.cmd == "list-variations":
        for v in m.list_variations(args.voice_id):
            exists = "" if os.path.exists(v["voice_ref_path"]) else "  [reference missing]"
            print(f"{v['variation_type']:10s} {v['voice_name']:32s} ${v.get('price_usd', 0.0):.2f}{exists}")
    elif args.cmd == "onboard-seller-payments":
        try:
            result = m.onboard_seller_payments(args.seller_id, args.email)
        except StripeNotConfiguredError as e:
            print(f"Stripe is not ready yet: {e}")
            return
        print(_json.dumps(result, indent=2))
        print("\nSend the seller to onboarding_url to finish connecting their Stripe account.")
    elif args.cmd == "seller-payment-status":
        try:
            print(_json.dumps(m.get_seller_payment_status(args.seller_id), indent=2))
        except StripeNotConfiguredError as e:
            print(f"Stripe is not ready yet: {e}")
    elif args.cmd == "create-checkout":
        try:
            result = m.create_purchase_checkout(args.voice_id, buyer_email=args.buyer_email, purpose=args.purpose)
        except StripeNotConfiguredError as e:
            print(f"Stripe is not ready yet: {e}")
            return
        print(_json.dumps(result, indent=2))
        print("\nSend the buyer to checkout_url to complete payment.")


if __name__ == "__main__":
    main()

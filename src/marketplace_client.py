#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTTP client for the standalone "Volcano Studios Voice Marketplace" product.

The Voice Marketplace used to live entirely inside this repo
(src/voice_marketplace.py + src/marketplace_payments.py, driving a local
Qdrant store). It has since been extracted into its own standalone product
(a separate FastAPI service, containerized via Docker) so it can be sold/run
independently of any one production tool. This module is the seam that lets
Firespeaker keep talking to marketplace listings, licenses, and Stripe
Connect/Checkout flows without caring that they now live behind a REST API
on a different process (possibly a different machine) instead of in-process.

MarketplaceClient is deliberately duck-type compatible with the original
VoiceMarketplace class -- same public method names and signatures -- so
existing call sites (gui_server.py, console_api.py, voice_dataset.py) need
little to no change beyond swapping which class their lazy singleton
constructs. The one place that ISN'T a thin proxy is cast_character() /
cast_character_with_voice(): binding a purchased voice to a character's
MemPalace "drawer" is Firespeaker-local business logic (it directly edits
this repo's local SQLite drawers table), so that logic is re-implemented
here on top of the remote purchase + audio-download primitives instead of
being forwarded to the server.

Configuration: MARKETPLACE_API_URL env var (default http://localhost:8010,
matching the standalone product's docker-compose published port). Loaded
from a .env file at the repo root via python-dotenv, same convention as
src/llm_client.py and src/marketplace_payments.py.
"""

import os
import re
import json
import base64
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MarketplaceClient")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    logger.warning("python-dotenv not installed; relying on process environment only.")

MARKETPLACE_API_URL = os.getenv("MARKETPLACE_API_URL", "http://localhost:8010").rstrip("/")

CACHE_DIR = os.path.join("data", "marketplace_cache")

# Re-exported (not redefined) so existing `except StripeNotConfiguredError`
# blocks in gui_server.py keep working unchanged even though the error now
# originates from an HTTP 503 response instead of an in-process Stripe call.
from src.marketplace_payments import StripeNotConfiguredError


class MarketplaceClient:
    """Duck-type replacement for src.voice_marketplace.VoiceMarketplace that
    talks to the standalone marketplace product over HTTP instead of a local
    Qdrant store."""

    def __init__(self, base_url: str = MARKETPLACE_API_URL, seller_id: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.default_seller_id = seller_id or "local"
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ----------------------------------------------------
    # Low-level HTTP plumbing
    # ----------------------------------------------------

    def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                detail = json.loads(raw).get("detail", raw.decode("utf-8", "replace"))
            except Exception:
                detail = raw.decode("utf-8", "replace") if raw else str(e)
            if e.code == 503:
                raise StripeNotConfiguredError(str(detail))
            if e.code in (400, 404, 422):
                raise ValueError(str(detail))
            raise RuntimeError(f"Marketplace request failed ({e.code}): {detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach the Voice Marketplace service at {self.base_url} ({e.reason}). "
                "Is the standalone marketplace container running? (docker compose up -d in the "
                "volcano-studios-voice-marketplace project)"
            )

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, json_body: Optional[Dict[str, Any]] = None,
             seller_id: Optional[str] = None) -> Dict[str, Any]:
        headers = {"X-Seller-Id": seller_id or self.default_seller_id}
        return self._request("POST", path, json_body=json_body, headers=headers)

    def _upload_local_file(self, local_path: str) -> str:
        """Uploads a local sample WAV to the remote marketplace's staging area
        and returns the *remote* server-side path onboard_voice/onboard_variation
        expect. Lets callers (voice_dataset.py, voice_studio.py) keep passing
        local file paths unchanged even though onboarding itself now happens
        on a different machine/container."""
        if not os.path.exists(local_path):
            raise ValueError(f"Sample not found: {local_path}")
        with open(local_path, "rb") as f:
            raw = f.read()
        data_url = "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")
        result = self._post("/api/marketplace/upload_sample", {
            "filename": os.path.basename(local_path),
            "data": data_url,
        })
        return result["path"]

    # ----------------------------------------------------
    # Browse / search / listing lookup
    # ----------------------------------------------------

    def list_all(self, limit: int = 200, include_variations: bool = True) -> List[Dict[str, Any]]:
        listings = self._get("/api/marketplace/listings")["listings"]
        if not include_variations:
            listings = [l for l in listings if not l.get("parent_voice_id")]
        return listings[:limit]

    def search_marketplace(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        from urllib.parse import urlencode
        qs = urlencode({"q": query, "limit": limit})
        return self._get(f"/api/marketplace/search?{qs}")["results"]

    def get_listing(self, voice_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._get(f"/api/marketplace/listing/{voice_id}")["listing"]
        except ValueError:
            return None

    def list_variations(self, parent_voice_id: str) -> List[Dict[str, Any]]:
        from urllib.parse import urlencode
        qs = urlencode({"voice_id": parent_voice_id})
        return self._get(f"/api/marketplace/variations?{qs}")["variations"]

    # ----------------------------------------------------
    # Seller onboarding
    # ----------------------------------------------------

    def onboard_voice(self, seller_name: str, voice_name: str, sample_wav_paths: List[str],
                      description: str, price_usd: float = 0.0, consent_confirmed: bool = False,
                      seller_id: str = "local") -> Dict[str, Any]:
        remote_samples = [self._upload_local_file(p) for p in sample_wav_paths]
        result = self._post("/api/marketplace/onboard", {
            "seller": seller_name,
            "name": voice_name,
            "samples": remote_samples,
            "description": description,
            "price": price_usd,
            "consent": consent_confirmed,
        }, seller_id=seller_id)
        return result["listing"]

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
        remote_samples = [self._upload_local_file(p) for p in sample_wav_paths] if sample_wav_paths else None
        result = self._post("/api/marketplace/onboard_variation", {
            "parent_voice_id": parent_voice_id,
            "label": variation_label,
            "description": description,
            "price": price_usd,
            "samples": remote_samples,
            "preset": algorithmic_preset,
            "pitch_semitones": pitch_semitones,
            "speed_ratio": speed_ratio,
        })
        return result["variation"]

    # ----------------------------------------------------
    # Purchase / licensing
    # ----------------------------------------------------

    def purchase_voice(self, voice_id: str, buyer: str, purpose: str,
                       payment_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal/free license grant. `payment_info` is accepted for
        signature compatibility with VoiceMarketplace.purchase_voice but is
        ignored -- real-money purchases now go through POST /api/marketplace/
        checkout followed by Stripe calling the standalone product's own
        webhook directly, never through this client."""
        result = self._post("/api/marketplace/purchase", {
            "voice_id": voice_id,
            "buyer": buyer,
            "purpose": purpose,
        })
        return result["license"]

    def list_licenses_for_buyer(self, buyer: str) -> List[Dict[str, Any]]:
        from urllib.parse import urlencode
        qs = urlencode({"buyer": buyer})
        return self._get(f"/api/marketplace/licenses?{qs}")["licenses"]

    # ----------------------------------------------------
    # Stripe Connect (seller payouts + buyer checkout)
    # ----------------------------------------------------

    def onboard_seller_payments(self, seller_id: str, email: str) -> Dict[str, Any]:
        return self._post("/api/marketplace/seller/connect", {"email": email}, seller_id=seller_id)["connect"]

    def get_seller_payment_status(self, seller_id: str) -> Dict[str, Any]:
        return self._get(f"/api/marketplace/seller/status?seller_id={seller_id}")["status"]

    def create_purchase_checkout(self, voice_id: str, buyer_email: Optional[str] = None,
                                 purpose: str = "") -> Dict[str, Any]:
        return self._post("/api/marketplace/checkout", {
            "voice_id": voice_id,
            "buyer_email": buyer_email,
            "purpose": purpose,
        })["checkout"]

    # ----------------------------------------------------
    # Audio download (with local caching)
    # ----------------------------------------------------

    def download_audio(self, voice_id: str) -> str:
        """Downloads (and caches under data/marketplace_cache/) the reference
        WAV for a listing, returning a local file path. Needed because the
        marketplace's audio files live on the standalone product's own
        filesystem now -- Firespeaker can no longer assume voice_ref_path
        from a listing payload points to a locally-reachable file."""
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", voice_id)
        cached_path = os.path.join(CACHE_DIR, f"{safe_id}.wav")
        if os.path.exists(cached_path):
            return cached_path
        url = f"{self.base_url}/api/marketplace/audio/{voice_id}"
        req = urllib.request.Request(url, headers={"Accept": "audio/wav"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"No reference audio available for listing: {voice_id}")
            raise RuntimeError(f"Failed to download audio for {voice_id}: HTTP {e.code}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach the Voice Marketplace service at {self.base_url} ({e.reason})")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cached_path, "wb") as f:
            f.write(raw)
        return cached_path

    # ----------------------------------------------------
    # Casting bridge (Firespeaker-local: binds a purchased voice to a
    # character's MemPalace drawer). Re-implemented here rather than proxied,
    # since it directly manipulates this repo's local SQLite drawers table.
    # ----------------------------------------------------

    def cast_character(self, character_name: str, character_description: str,
                       buyer: str = "local", purpose: str = "") -> Optional[Dict[str, Any]]:
        hits = self.search_marketplace(character_description, limit=3)
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
        license_record = self.purchase_voice(best["voice_id"], buyer=buyer, purpose=purpose or f"cast as {character_name}")
        local_ref_path = self.download_audio(best["voice_id"])

        from src.spatial_memory import MemPalace
        palace = MemPalace(use_chroma=False)
        try:
            if not palace.get_character_drawer(character_name):
                palace.register_character(character_name=character_name, voice_ref_path=local_ref_path, speed=1.0, pitch=0.0)
            else:
                drawer = palace.get_character_drawer(character_name)
                config = drawer["modulation_config"]
                config.pop("xtts_speaker", None)  # purchased reference beats builtin pin
                palace.conn.execute(
                    "UPDATE drawers SET voice_ref_path = ?, modulation_config_json = ? WHERE character_name = ?",
                    (local_ref_path, json.dumps(config), character_name),
                )
                palace.conn.commit()
        finally:
            palace.close()
        score_note = f" (score {best['score']:.2f})" if best.get("score") is not None else ""
        logger.info(f"Cast '{character_name}' with marketplace voice '{best['voice_name']}'{score_note}")
        best_with_local_path = dict(best)
        best_with_local_path["voice_ref_path"] = local_ref_path
        return {"character": character_name, "voice": best_with_local_path, "license": license_record}

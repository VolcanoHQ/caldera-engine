"""Integration tests for the /api/marketplace/* REST surface in src/gui_server.py.

Spins up a real ThreadingHTTPServer on a random port (no mocking of the HTTP
stack) and drives it with real HTTP requests, since StudioRequestHandler reads
directly from self.rfile/self.headers -- the raw-body Stripe webhook carve-out
in do_POST() in particular can't be verified against a mocked request object.
"""
import json
import base64
import threading
import urllib.request
import urllib.error

import pytest

from http.server import ThreadingHTTPServer

from src import gui_server
from src.voice_marketplace import VoiceMarketplace


def _fake_denoise_and_concat(sample_wav_paths, out_dir, out_name="reference_mono.wav"):
    """Stand-in for the real ffmpeg-based denoise/concat pipeline, which isn't
    available in this dev sandbox -- just copies the first sample through so
    the REST plumbing (not the audio pipeline) is what's under test here."""
    import os
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    shutil.copy(sample_wav_paths[0], out_path)
    return out_path, 1.0


def _fake_render_algorithmic_variant(source_wav, out_dir, params):
    import os
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "variant.wav")
    shutil.copy(source_wav, out_path)
    return out_path


@pytest.fixture()
def server(tmp_path, monkeypatch):
    # Isolated Qdrant db per test, and make sure no real Stripe creds leak in
    # from a developer's local .env during the test run.
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    mp = VoiceMarketplace(db_path=str(tmp_path / "qdrant_db"))
    # ffmpeg isn't available in this dev sandbox -- swap the audio-processing
    # boundary for fakes, same pattern as tests/test_voice_marketplace_variations.py.
    mp._denoise_and_concat = _fake_denoise_and_concat
    mp._render_algorithmic_variant = _fake_render_algorithmic_variant
    gui_server.StudioRequestHandler._marketplace = mp

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gui_server.StudioRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        gui_server.StudioRequestHandler._marketplace = None


def _get(base_url, path):
    req = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _post(base_url, path, payload, extra_headers=None, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _make_sample_wav(tmp_path, name="sample.wav"):
    import wave
    import struct
    path = tmp_path / name
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = b"".join(struct.pack("<h", 0) for _ in range(16000 * 2))
        w.writeframes(frames)
    return str(path)


def test_onboard_and_list_listings(server, tmp_path):
    sample = _make_sample_wav(tmp_path)
    status, body = _post(server, "/api/marketplace/onboard", {
        "seller": "Jane Seller",
        "name": "Warm Narrator",
        "samples": [sample],
        "description": "A warm, calm narrator voice",
        "price": 25.0,
        "consent": True,
    })
    assert status == 200, body
    voice_id = body["listing"]["voice_id"]
    assert voice_id

    status, body = _get(server, "/api/marketplace/listings")
    assert status == 200
    assert any(l["voice_id"] == voice_id for l in body["listings"])


def test_onboard_variation_and_list_variations(server, tmp_path):
    sample = _make_sample_wav(tmp_path)
    status, body = _post(server, "/api/marketplace/onboard", {
        "seller": "Jane Seller",
        "name": "Variation Base",
        "samples": [sample],
        "description": "Base voice for variation test",
        "price": 10.0,
        "consent": True,
    })
    assert status == 200, body
    voice_id = body["listing"]["voice_id"]

    status, body = _post(server, "/api/marketplace/onboard_variation", {
        "parent_voice_id": voice_id,
        "label": "excited",
        "description": "An excited, upbeat delivery",
        "price": 5.0,
        "preset": "bright_fast",
    })
    assert status == 200, body
    assert body["variation"]["parent_voice_id"] == voice_id
    assert body["variation"]["variation_type"] == "algorithmic"

    status, body = _get(server, f"/api/marketplace/variations?voice_id={voice_id}")
    assert status == 200
    assert len(body["variations"]) == 1
    assert body["variations"][0]["variation_label"] == "excited"


def test_variations_missing_voice_id_400(server):
    status, body = _get(server, "/api/marketplace/variations")
    assert status == 400


def test_seller_connect_and_status_without_real_stripe_key_returns_503(server):
    status, body = _post(server, "/api/marketplace/seller/connect", {"email": "seller@example.com"})
    assert status == 503
    assert "error" in body

    # refresh_seller_account_status also short-circuits on StripeNotConfiguredError
    # before it would ever get to "no account on file" -> also 503, not 400.
    status, body = _get(server, "/api/marketplace/seller/status?seller_id=local")
    assert status == 503
    assert "error" in body


def test_checkout_without_real_stripe_key_returns_503(server, tmp_path):
    sample = _make_sample_wav(tmp_path)
    status, body = _post(server, "/api/marketplace/onboard", {
        "seller": "Jane Seller",
        "name": "Checkout Test Voice",
        "samples": [sample],
        "description": "Voice for checkout REST test",
        "price": 12.0,
        "consent": True,
    })
    voice_id = body["listing"]["voice_id"]

    status, body = _post(server, "/api/marketplace/checkout", {
        "voice_id": voice_id,
        "buyer_email": "buyer@example.com",
        "purpose": "test purchase",
    })
    assert status == 503
    assert "error" in body


def test_purchase_free_internal_grant_still_works(server, tmp_path):
    sample = _make_sample_wav(tmp_path)
    status, body = _post(server, "/api/marketplace/onboard", {
        "seller": "Jane Seller",
        "name": "Free Grant Voice",
        "samples": [sample],
        "description": "Voice for internal free purchase REST test",
        "price": 0.0,
        "consent": True,
    })
    voice_id = body["listing"]["voice_id"]

    status, body = _post(server, "/api/marketplace/purchase", {
        "voice_id": voice_id,
        "buyer": "local",
        "purpose": "internal casting",
    })
    assert status == 200, body
    assert body["license"]["payment"] is None


def test_webhook_endpoint_no_longer_proxied(server):
    # The Stripe webhook is no longer proxied through gui_server.py -- Stripe
    # is now configured to call the standalone Volcano Studios Voice
    # Marketplace product's own POST /api/marketplace/webhook/stripe
    # endpoint directly (see src/marketplace_client.py). This route no
    # longer exists on this server at all.
    status, body = _post(
        server, "/api/marketplace/webhook/stripe", None,
        extra_headers={"Stripe-Signature": "t=1,v1=bogus"},
        raw=b'{"id": "evt_test", "type": "checkout.session.completed"}',
    )
    assert status == 404


def test_unknown_endpoint_404(server):
    status, body = _get(server, "/api/marketplace/does_not_exist")
    assert status == 404

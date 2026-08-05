"""Unit tests for src/marketplace_client.py -- the HTTP client that replaced
the in-process src/voice_marketplace.py after the Voice Marketplace was
extracted into its own standalone product. No real HTTP calls are made:
urllib.request.urlopen is monkeypatched to simulate the standalone product's
JSON responses, matching its real REST contract (see
volcano-studios-voice-marketplace/src/volcano_marketplace/routers/*.py).
"""
import io
import json
import base64
import urllib.error

import pytest

from src.marketplace_client import MarketplaceClient
from src.marketplace_payments import StripeNotConfiguredError


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _json_response(payload):
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


def _http_error(code, detail):
    body = json.dumps({"detail": detail}).encode("utf-8")
    return urllib.error.HTTPError(url="http://x", code=code, msg=detail, hdrs=None, fp=io.BytesIO(body))


@pytest.fixture()
def client():
    return MarketplaceClient(base_url="http://fake-marketplace:8010", seller_id="local")


def test_list_all(client, monkeypatch):
    listings = [{"voice_id": "abc", "voice_name": "Test Voice"}]

    def fake_urlopen(req, timeout=60):
        assert req.full_url == "http://fake-marketplace:8010/api/marketplace/listings"
        return _json_response({"listings": listings})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert client.list_all() == listings


def test_get_listing_404_returns_none(client, monkeypatch):
    def fake_urlopen(req, timeout=60):
        raise _http_error(404, "No such voice listing: nope")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert client.get_listing("nope") is None


def test_get_listing_found(client, monkeypatch):
    def fake_urlopen(req, timeout=60):
        return _json_response({"listing": {"voice_id": "abc", "voice_name": "Test"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    listing = client.get_listing("abc")
    assert listing["voice_id"] == "abc"


def test_checkout_503_raises_stripe_not_configured(client, monkeypatch):
    def fake_urlopen(req, timeout=60):
        raise _http_error(503, "STRIPE_SECRET_KEY is not set.")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(StripeNotConfiguredError):
        client.create_purchase_checkout("voice-1")


def test_purchase_bad_voice_id_raises_value_error(client, monkeypatch):
    def fake_urlopen(req, timeout=60):
        raise _http_error(400, "No such voice listing: bogus")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        client.purchase_voice("bogus", buyer="local", purpose="test")


def test_onboard_voice_uploads_then_onboards(client, monkeypatch, tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFF....WAVEfmt ")

    calls = []

    def fake_urlopen(req, timeout=60):
        calls.append(req.full_url)
        if req.full_url.endswith("/upload_sample"):
            sent = json.loads(req.data.decode("utf-8"))
            assert sent["data"].startswith("data:audio/wav;base64,")
            assert base64.b64decode(sent["data"].split(",", 1)[1]) == b"RIFF....WAVEfmt "
            return _json_response({"path": "/remote/uploads/sample.wav", "bytes": 16})
        assert req.full_url.endswith("/onboard")
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["samples"] == ["/remote/uploads/sample.wav"]
        assert req.headers.get("X-seller-id") == "local"
        return _json_response({"listing": {"voice_id": "new-voice", "voice_name": sent["name"]}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    listing = client.onboard_voice(
        seller_name="Jane", voice_name="Warm Narrator", sample_wav_paths=[str(sample)],
        description="A warm narrator", price_usd=10.0, consent_confirmed=True,
    )
    assert listing["voice_id"] == "new-voice"
    assert calls[0].endswith("/upload_sample")
    assert calls[1].endswith("/onboard")


def test_download_audio_caches_locally(client, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=60):
        call_count["n"] += 1
        assert req.full_url == "http://fake-marketplace:8010/api/marketplace/audio/voice-1"
        return _FakeResponse(b"FAKEWAVBYTES")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    path1 = client.download_audio("voice-1")
    assert open(path1, "rb").read() == b"FAKEWAVBYTES"
    # Second call should hit the local cache, not urlopen again.
    path2 = client.download_audio("voice-1")
    assert path1 == path2
    assert call_count["n"] == 1


def test_cast_character_with_voice_binds_mempalace_drawer(client, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_urlopen(req, timeout=60):
        if "/api/marketplace/purchase" in req.full_url:
            return _json_response({"license": {"license_id": "lic-1", "voice_id": "voice-1"}})
        if "/api/marketplace/audio/" in req.full_url:
            return _FakeResponse(b"FAKEWAVBYTES")
        raise AssertionError(f"unexpected request: {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    registered = {}

    class _FakePalace:
        def __init__(self, use_chroma=False):
            pass

        def get_character_drawer(self, name):
            return registered.get(name)

        def register_character(self, character_name, voice_ref_path, speed=1.0, pitch=0.0):
            registered[character_name] = {"voice_ref_path": voice_ref_path, "modulation_config": {}}

        def close(self):
            pass

    import src.spatial_memory as spatial_memory
    monkeypatch.setattr(spatial_memory, "MemPalace", _FakePalace)

    result = client.cast_character_with_voice(
        character_name="Captain Ahab",
        voice_id="voice-1",
        buyer="local",
        purpose="test cast",
        preselected_voice={"voice_id": "voice-1", "voice_name": "Old Salt Captain"},
    )
    assert result["license"]["license_id"] == "lic-1"
    assert "Captain Ahab" in registered
    assert registered["Captain Ahab"]["voice_ref_path"].endswith(".wav")

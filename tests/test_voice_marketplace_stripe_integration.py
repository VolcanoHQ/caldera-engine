#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the VoiceMarketplace <-> Stripe Connect integration methods
(onboard_seller_payments, get_seller_payment_status, create_purchase_checkout,
confirm_stripe_webhook). These use a real (embedded, tmp_path) Qdrant instance
for listing storage, but a fake MarketplacePayments stand-in for the actual
Stripe network calls -- Stripe itself is covered in
tests/test_marketplace_payments.py.
"""

import pytest

from src.voice_marketplace import VoiceMarketplace


class _FakePayments:
    """Stands in for MarketplacePayments so these tests exercise only the
    VoiceMarketplace-side plumbing (listing lookups, license grants) without
    touching Stripe or the real seller ledger file."""

    def __init__(self):
        self.seller_accounts = {}
        self.checkout_calls = []
        self.webhook_events = []

    def create_seller_connect_account(self, seller_id, email):
        self.seller_accounts[seller_id] = {"email": email, "charges_enabled": False}
        return {"seller_id": seller_id, "stripe_account_id": "acct_fake", "onboarding_url": "https://fake/onboard"}

    def refresh_seller_account_status(self, seller_id):
        record = self.seller_accounts.setdefault(seller_id, {})
        record["charges_enabled"] = True
        return record

    def create_checkout_session(self, voice_id, voice_name, price_usd, seller_id, buyer_email=None, purpose=""):
        self.checkout_calls.append(dict(
            voice_id=voice_id, voice_name=voice_name, price_usd=price_usd,
            seller_id=seller_id, buyer_email=buyer_email, purpose=purpose,
        ))
        return {"session_id": "cs_test_1", "checkout_url": "https://fake/checkout", "amount_usd": price_usd, "platform_fee_usd": price_usd * 0.15}


@pytest.fixture
def marketplace(tmp_path):
    m = VoiceMarketplace(db_path=str(tmp_path / "qdrant_db"))
    m._payments = _FakePayments()
    yield m
    m.client.close()


def test_onboard_seller_payments_delegates_to_payments_client(marketplace):
    result = marketplace.onboard_seller_payments("seller-1", "seller@example.com")
    assert result["onboarding_url"] == "https://fake/onboard"
    assert "seller-1" in marketplace._payments.seller_accounts


def test_get_seller_payment_status_delegates_to_payments_client(marketplace):
    marketplace.onboard_seller_payments("seller-1", "seller@example.com")
    status = marketplace.get_seller_payment_status("seller-1")
    assert status["charges_enabled"] is True


def test_create_purchase_checkout_rejects_unknown_listing(marketplace):
    with pytest.raises(ValueError, match="No such voice listing"):
        marketplace.create_purchase_checkout("nonexistent-voice")


def test_create_purchase_checkout_rejects_listing_without_seller_id(marketplace):
    voice_id = marketplace.register_voice("Legacy Voice", "refs/legacy.wav", "a plain narrator")
    with pytest.raises(ValueError, match="no seller_id on file"):
        marketplace.create_purchase_checkout(voice_id)


def test_create_purchase_checkout_passes_listing_details_to_payments_client(marketplace, monkeypatch):
    voice_id = marketplace.register_voice("Storyteller", "refs/storyteller.wav", "a warm storyteller voice")
    marketplace.client.set_payload(
        collection_name="voice_marketplace",
        payload={"seller_id": "seller-1", "price_usd": 25.0},
        points=[voice_id],
    )

    result = marketplace.create_purchase_checkout(voice_id, buyer_email="buyer@example.com", purpose="cast as Narrator")

    assert result["checkout_url"] == "https://fake/checkout"
    call = marketplace._payments.checkout_calls[0]
    assert call["voice_id"] == voice_id
    assert call["voice_name"] == "Storyteller"
    assert call["price_usd"] == 25.0
    assert call["seller_id"] == "seller-1"
    assert call["buyer_email"] == "buyer@example.com"
    assert call["purpose"] == "cast as Narrator"


def test_confirm_stripe_webhook_grants_license_for_paid_checkout(marketplace, monkeypatch):
    voice_id = marketplace.register_voice("Storyteller", "refs/storyteller.wav", "a warm storyteller voice")
    marketplace.client.set_payload(
        collection_name="voice_marketplace",
        payload={"seller_id": "seller-1", "price_usd": 25.0},
        points=[voice_id],
    )

    event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "payment_status": "paid",
                "id": "cs_test_abc",
                "payment_intent": "pi_test_abc",
                "amount_total": 2500,
                "metadata": {
                    "voice_id": voice_id,
                    "seller_id": "seller-1",
                    "buyer_email": "buyer@example.com",
                    "purpose": "cast as Narrator",
                },
            }
        },
    }
    marketplace._payments.construct_webhook_event = lambda payload, sig_header: event

    license_record = marketplace.confirm_stripe_webhook(b"raw-payload", "fake-sig")

    assert license_record is not None
    assert license_record["voice_id"] == voice_id
    assert license_record["buyer"] == "buyer@example.com"
    assert license_record["payment"]["stripe_session_id"] == "cs_test_abc"
    assert license_record["payment"]["stripe_payment_intent_id"] == "pi_test_abc"
    assert license_record["payment"]["amount_total_usd"] == 25.0


def test_confirm_stripe_webhook_returns_none_for_unrelated_event(marketplace):
    event = {"type": "payment_intent.created", "data": {"object": {}}}
    marketplace._payments.construct_webhook_event = lambda payload, sig_header: event

    assert marketplace.confirm_stripe_webhook(b"raw-payload", "fake-sig") is None

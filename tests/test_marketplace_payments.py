#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for src/marketplace_payments.py -- Stripe Connect seller onboarding,
buyer Checkout, and webhook-confirmed license grants.

Stripe itself is never called for real: these tests monkeypatch the `stripe`
module object that marketplace_payments imports (Account/AccountLink/
checkout.Session/Webhook), so the tests exercise this module's own logic
(config validation, seller ledger persistence, fee math, metadata plumbing)
without any network access or real/test Stripe credentials.
"""

import json
import types

import pytest

import src.marketplace_payments as mp


class _FakeAccount(dict):
    """Stripe SDK objects support both attribute-less dict access (obj["id"])
    and .get(...) -- a plain dict satisfies both call sites in our code."""


@pytest.fixture
def seller_ledger_path(tmp_path):
    return str(tmp_path / "seller_accounts.json")


@pytest.fixture
def payments(monkeypatch, seller_ledger_path):
    """A MarketplacePayments configured with a fake (non-placeholder) key and a
    fake `stripe` module, isolated to a tmp_path seller ledger."""
    monkeypatch.setattr(mp, "HAS_STRIPE", True)
    monkeypatch.setattr(mp, "STRIPE_SECRET_KEY", "sk_test_fake_for_testing")

    fake_stripe = types.SimpleNamespace()
    fake_stripe.api_key = None
    fake_stripe.Account = types.SimpleNamespace(
        create=lambda **kwargs: _FakeAccount(id="acct_fake123"),
        retrieve=lambda account_id: _FakeAccount(
            id=account_id, charges_enabled=True, payouts_enabled=True, details_submitted=True
        ),
    )
    fake_stripe.AccountLink = types.SimpleNamespace(
        create=lambda **kwargs: _FakeAccount(url="https://connect.stripe.com/setup/fake")
    )
    fake_stripe.checkout = types.SimpleNamespace(
        Session=types.SimpleNamespace(
            create=lambda **kwargs: _FakeAccount(
                id="cs_test_fake123",
                url="https://checkout.stripe.com/pay/cs_test_fake123",
                **{},
            )
        )
    )
    fake_stripe.Webhook = types.SimpleNamespace(
        construct_event=lambda payload, sig_header, secret: json.loads(payload)
    )
    monkeypatch.setattr(mp, "stripe", fake_stripe)

    return mp.MarketplacePayments(seller_ledger_path=seller_ledger_path)


def test_require_stripe_raises_when_key_is_placeholder(monkeypatch, seller_ledger_path):
    monkeypatch.setattr(mp, "HAS_STRIPE", True)
    monkeypatch.setattr(mp, "STRIPE_SECRET_KEY", "sk_test_placeholder")
    payments = mp.MarketplacePayments(seller_ledger_path=seller_ledger_path)

    with pytest.raises(mp.StripeNotConfiguredError, match="STRIPE_SECRET_KEY"):
        payments.create_seller_connect_account("seller-1", "seller@example.com")


def test_require_stripe_raises_when_package_missing(monkeypatch, seller_ledger_path):
    monkeypatch.setattr(mp, "HAS_STRIPE", False)
    payments = mp.MarketplacePayments(seller_ledger_path=seller_ledger_path)

    with pytest.raises(mp.StripeNotConfiguredError, match="pip install stripe"):
        payments.create_seller_connect_account("seller-1", "seller@example.com")


def test_is_live_ready_false_for_placeholder_key(monkeypatch, seller_ledger_path):
    monkeypatch.setattr(mp, "HAS_STRIPE", True)
    monkeypatch.setattr(mp, "STRIPE_SECRET_KEY", "sk_test_placeholder")
    payments = mp.MarketplacePayments(seller_ledger_path=seller_ledger_path)
    assert payments.is_live_ready() is False


def test_is_live_ready_true_for_real_looking_key(monkeypatch, seller_ledger_path):
    monkeypatch.setattr(mp, "HAS_STRIPE", True)
    monkeypatch.setattr(mp, "STRIPE_SECRET_KEY", "sk_live_realkey")
    payments = mp.MarketplacePayments(seller_ledger_path=seller_ledger_path)
    assert payments.is_live_ready() is True


def test_create_seller_connect_account_persists_and_returns_onboarding_url(payments, seller_ledger_path):
    result = payments.create_seller_connect_account("seller-1", "seller@example.com")

    assert result["seller_id"] == "seller-1"
    assert result["stripe_account_id"] == "acct_fake123"
    assert result["onboarding_url"] == "https://connect.stripe.com/setup/fake"

    with open(seller_ledger_path) as f:
        ledger = json.load(f)
    assert ledger["seller-1"]["stripe_account_id"] == "acct_fake123"
    assert ledger["seller-1"]["email"] == "seller@example.com"
    assert ledger["seller-1"]["charges_enabled"] is False


def test_create_seller_connect_account_reuses_existing_account(payments, seller_ledger_path, monkeypatch):
    payments.create_seller_connect_account("seller-1", "seller@example.com")

    calls = []
    payments_stripe = mp.stripe
    monkeypatch.setattr(
        payments_stripe.Account, "create",
        lambda **kwargs: calls.append(kwargs) or _FakeAccount(id="acct_should_not_be_used"),
    )

    result = payments.create_seller_connect_account("seller-1", "seller@example.com")
    assert result["stripe_account_id"] == "acct_fake123"
    assert calls == []  # Account.create should not be called again for an existing seller


def test_refresh_seller_account_status_updates_ledger(payments, seller_ledger_path):
    payments.create_seller_connect_account("seller-1", "seller@example.com")

    status = payments.refresh_seller_account_status("seller-1")
    assert status["charges_enabled"] is True
    assert status["payouts_enabled"] is True
    assert status["details_submitted"] is True

    with open(seller_ledger_path) as f:
        ledger = json.load(f)
    assert ledger["seller-1"]["charges_enabled"] is True


def test_refresh_seller_account_status_rejects_unknown_seller(payments):
    with pytest.raises(ValueError, match="No Stripe Connect account"):
        payments.refresh_seller_account_status("nonexistent-seller")


def test_create_checkout_session_rejects_zero_price(payments):
    with pytest.raises(ValueError, match="greater than zero"):
        payments.create_checkout_session(
            voice_id="v1", voice_name="Test Voice", price_usd=0.0, seller_id="seller-1"
        )


def test_create_checkout_session_rejects_seller_without_stripe_account(payments):
    with pytest.raises(ValueError, match="not connected a Stripe account"):
        payments.create_checkout_session(
            voice_id="v1", voice_name="Test Voice", price_usd=25.0, seller_id="unknown-seller"
        )


def test_create_checkout_session_rejects_seller_without_charges_enabled(payments):
    payments.create_seller_connect_account("seller-1", "seller@example.com")
    # Freshly created accounts start with charges_enabled=False until onboarding completes.
    with pytest.raises(ValueError, match="not finished Stripe onboarding"):
        payments.create_checkout_session(
            voice_id="v1", voice_name="Test Voice", price_usd=25.0, seller_id="seller-1"
        )


def test_create_checkout_session_computes_platform_fee(payments, monkeypatch):
    payments.create_seller_connect_account("seller-1", "seller@example.com")
    payments.refresh_seller_account_status("seller-1")
    monkeypatch.setattr(mp, "MARKETPLACE_PLATFORM_FEE_PERCENT", 15.0)

    captured = {}
    monkeypatch.setattr(
        mp.stripe.checkout.Session, "create",
        lambda **kwargs: captured.update(kwargs) or _FakeAccount(id="cs_test_1", url="https://checkout.stripe.com/pay/cs_test_1"),
    )

    result = payments.create_checkout_session(
        voice_id="v1", voice_name="Test Voice", price_usd=20.0, seller_id="seller-1", buyer_email="buyer@example.com", purpose="cast as Narrator"
    )

    assert result["checkout_url"] == "https://checkout.stripe.com/pay/cs_test_1"
    assert result["amount_usd"] == pytest.approx(20.0)
    assert result["platform_fee_usd"] == pytest.approx(3.0)  # 15% of $20
    assert captured["payment_intent_data"]["application_fee_amount"] == 300  # cents
    assert captured["payment_intent_data"]["transfer_data"]["destination"] == "acct_fake123"
    assert captured["metadata"]["voice_id"] == "v1"
    assert captured["metadata"]["seller_id"] == "seller-1"
    assert captured["metadata"]["purpose"] == "cast as Narrator"


def test_extract_completed_checkout_metadata_ignores_other_event_types():
    event = {"type": "payment_intent.created", "data": {"object": {}}}
    assert mp.MarketplacePayments.extract_completed_checkout_metadata(event) is None


def test_extract_completed_checkout_metadata_ignores_unpaid_sessions():
    event = {
        "type": "checkout.session.completed",
        "data": {"object": {"payment_status": "unpaid", "metadata": {}}},
    }
    assert mp.MarketplacePayments.extract_completed_checkout_metadata(event) is None


def test_extract_completed_checkout_metadata_returns_license_fields_for_paid_session():
    event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "payment_status": "paid",
                "id": "cs_test_abc",
                "payment_intent": "pi_test_abc",
                "amount_total": 2500,
                "metadata": {
                    "voice_id": "v1",
                    "seller_id": "seller-1",
                    "buyer_email": "buyer@example.com",
                    "purpose": "cast as Narrator",
                },
            }
        },
    }
    metadata = mp.MarketplacePayments.extract_completed_checkout_metadata(event)
    assert metadata == {
        "voice_id": "v1",
        "seller_id": "seller-1",
        "buyer_email": "buyer@example.com",
        "purpose": "cast as Narrator",
        "stripe_session_id": "cs_test_abc",
        "stripe_payment_intent_id": "pi_test_abc",
        "amount_total_usd": 25.0,
    }


def test_construct_webhook_event_delegates_to_stripe_webhook(payments):
    payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {}}}).encode()
    event = payments.construct_webhook_event(payload, "fake_sig_header")
    assert event["type"] == "checkout.session.completed"

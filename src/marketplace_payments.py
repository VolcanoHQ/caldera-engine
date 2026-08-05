#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Voice Marketplace -- Stripe Connect payments.

*** DEPRECATED as the live backing implementation. ***
The Voice Marketplace (and its Stripe Connect payment handling) has been
extracted into its own standalone product ("Volcano Studios Voice
Marketplace" -- separate repo, runs via Docker). This app now talks to it
over HTTP via src/marketplace_client.py, and Stripe is configured to call
that product's own webhook endpoint directly -- not this module. This file
is kept only because tests/test_marketplace_payments.py and
tests/test_voice_marketplace_stripe_integration.py still exercise it
directly; it is not imported by any live request-handling code path anymore.

Handles the real-money side of the marketplace: sellers connect a Stripe
Express account to receive payouts, buyers pay through Stripe Checkout, and a
webhook confirms the charge before a license is granted. Voice licensing
records/casting logic itself stays in voice_marketplace.py -- this module is
only responsible for moving money and reporting seller payout state.

All configuration is read from environment variables (loaded from a .env file
at the repo root via python-dotenv, same convention as src/llm_client.py) so
placeholder/test-mode keys can be swapped for live keys later with no code
changes. See .env.example for the full list of variables this module reads.
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MarketplacePayments")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
SELLER_LEDGER_PATH = os.path.join(REPO_ROOT, "data", "voice_marketplace", "seller_accounts.json")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    logger.warning("python-dotenv not installed; relying on process environment only.")

# Graceful optional dependency loading, matching the pattern used for the real
# text-embedding model in voice_marketplace.py: the marketplace still imports
# and runs everywhere, it just can't move real money until `stripe` is
# installed and a real (non-placeholder) secret key is configured.
HAS_STRIPE = False
try:
    import stripe
    HAS_STRIPE = True
except ImportError:
    logger.warning("stripe package not installed; Voice Marketplace payments are disabled until it is installed.")

# --- Configuration (all placeholder-friendly; set real values in .env) ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "sk_test_placeholder")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "pk_test_placeholder")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_placeholder")
MARKETPLACE_CURRENCY = os.getenv("MARKETPLACE_CURRENCY", "usd").lower()
# Platform's cut of each sale, taken as a Stripe Connect application fee.
MARKETPLACE_PLATFORM_FEE_PERCENT = float(os.getenv("MARKETPLACE_PLATFORM_FEE_PERCENT", "15"))
MARKETPLACE_CHECKOUT_SUCCESS_URL = os.getenv(
    "MARKETPLACE_CHECKOUT_SUCCESS_URL", "http://localhost:8420/marketplace/purchase-success?session_id={CHECKOUT_SESSION_ID}"
)
MARKETPLACE_CHECKOUT_CANCEL_URL = os.getenv(
    "MARKETPLACE_CHECKOUT_CANCEL_URL", "http://localhost:8420/marketplace/purchase-cancelled"
)
MARKETPLACE_CONNECT_RETURN_URL = os.getenv(
    "MARKETPLACE_CONNECT_RETURN_URL", "http://localhost:8420/marketplace/seller-onboarded"
)
MARKETPLACE_CONNECT_REFRESH_URL = os.getenv(
    "MARKETPLACE_CONNECT_REFRESH_URL", "http://localhost:8420/marketplace/seller-onboarding-refresh"
)

_PLACEHOLDER_KEYS = {"sk_test_placeholder", "", None}


class StripeNotConfiguredError(RuntimeError):
    """Raised when a real Stripe operation is attempted without a real secret key
    or without the `stripe` package installed. Distinct from ValueError (bad
    caller input) so REST layers can map it to a clear 501/503 instead of a 400."""


class MarketplacePayments:
    """Stripe Connect integration for the Voice Marketplace.

    Sellers get a Stripe Express connected account (onboarding handled entirely
    by Stripe's hosted flow via an Account Link -- we never touch bank details).
    Buyers pay via a Stripe Checkout Session that is a "destination charge":
    the full price is charged to the buyer, Stripe automatically transfers
    (price - platform fee) to the seller's connected account, and the platform
    fee stays on the platform's own Stripe balance.
    """

    def __init__(self, seller_ledger_path: str = SELLER_LEDGER_PATH):
        self.seller_ledger_path = seller_ledger_path
        if HAS_STRIPE:
            stripe.api_key = STRIPE_SECRET_KEY

    # ------------------------------------------------------------------
    # Config / readiness
    # ------------------------------------------------------------------

    def is_live_ready(self) -> bool:
        """True only once a real secret key is configured (not the placeholder)
        and the stripe package is importable. Used to give callers/UI a clear
        "test mode" vs "ready to take real payments" signal."""
        return HAS_STRIPE and STRIPE_SECRET_KEY not in _PLACEHOLDER_KEYS and not STRIPE_SECRET_KEY.startswith("sk_test_placeholder")

    def _require_stripe(self):
        if not HAS_STRIPE:
            raise StripeNotConfiguredError(
                "The 'stripe' package is not installed. Run: pip install stripe"
            )
        if STRIPE_SECRET_KEY in _PLACEHOLDER_KEYS:
            raise StripeNotConfiguredError(
                "STRIPE_SECRET_KEY is not set. Add it to your .env file (see .env.example)."
            )

    # ------------------------------------------------------------------
    # Seller ledger (seller_id -> Stripe Connect account bookkeeping)
    # ------------------------------------------------------------------

    def _load_seller_ledger(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.seller_ledger_path):
            return {}
        with open(self.seller_ledger_path) as f:
            return json.load(f)

    def _save_seller_ledger(self, ledger: Dict[str, Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.seller_ledger_path), exist_ok=True)
        with open(self.seller_ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)

    def get_seller_account_record(self, seller_id: str) -> Optional[Dict[str, Any]]:
        return self._load_seller_ledger().get(seller_id)

    # ------------------------------------------------------------------
    # Seller onboarding (Stripe Express Connect)
    # ------------------------------------------------------------------

    def create_seller_connect_account(self, seller_id: str, email: str) -> Dict[str, Any]:
        """Creates (or reuses) a Stripe Express connected account for a seller and
        returns a one-time hosted onboarding link. Stripe collects identity/bank
        details directly -- the marketplace only ever stores the account id."""
        self._require_stripe()

        ledger = self._load_seller_ledger()
        record = ledger.get(seller_id)

        if record and record.get("stripe_account_id"):
            account_id = record["stripe_account_id"]
        else:
            account = stripe.Account.create(
                type="express",
                email=email,
                capabilities={
                    "card_payments": {"requested": True},
                    "transfers": {"requested": True},
                },
            )
            account_id = account["id"]
            record = {
                "stripe_account_id": account_id,
                "email": email,
                "charges_enabled": False,
                "payouts_enabled": False,
                "details_submitted": False,
                "created_at": time.time(),
            }
            ledger[seller_id] = record
            self._save_seller_ledger(ledger)

        account_link = stripe.AccountLink.create(
            account=account_id,
            refresh_url=MARKETPLACE_CONNECT_REFRESH_URL,
            return_url=MARKETPLACE_CONNECT_RETURN_URL,
            type="account_onboarding",
        )
        return {
            "seller_id": seller_id,
            "stripe_account_id": account_id,
            "onboarding_url": account_link["url"],
        }

    def refresh_seller_account_status(self, seller_id: str) -> Dict[str, Any]:
        """Pulls the latest charges_enabled/payouts_enabled/details_submitted
        flags from Stripe and persists them, so sellers can see (and the
        purchase flow can enforce) whether payouts are actually live yet."""
        self._require_stripe()

        ledger = self._load_seller_ledger()
        record = ledger.get(seller_id)
        if not record:
            raise ValueError(f"No Stripe Connect account on file for seller: {seller_id}")

        account = stripe.Account.retrieve(record["stripe_account_id"])
        record.update({
            "charges_enabled": bool(account.get("charges_enabled")),
            "payouts_enabled": bool(account.get("payouts_enabled")),
            "details_submitted": bool(account.get("details_submitted")),
            "updated_at": time.time(),
        })
        ledger[seller_id] = record
        self._save_seller_ledger(ledger)
        return record

    # ------------------------------------------------------------------
    # Buyer checkout (destination charge with a platform application fee)
    # ------------------------------------------------------------------

    def create_checkout_session(
        self,
        voice_id: str,
        voice_name: str,
        price_usd: float,
        seller_id: str,
        buyer_email: Optional[str] = None,
        purpose: str = "",
    ) -> Dict[str, Any]:
        """Creates a Stripe Checkout Session that charges the buyer the full
        listing price and routes (price - platform fee) to the seller's
        connected account as a destination charge. The voice_id/seller_id/
        buyer/purpose are stashed in session metadata so the webhook handler
        can grant the license without re-deriving anything."""
        self._require_stripe()
        if price_usd <= 0:
            raise ValueError("price_usd must be greater than zero to check out (free voices don't need Stripe).")

        seller_record = self.get_seller_account_record(seller_id)
        if not seller_record or not seller_record.get("stripe_account_id"):
            raise ValueError(f"Seller '{seller_id}' has not connected a Stripe account yet.")
        if not seller_record.get("charges_enabled"):
            raise ValueError(
                f"Seller '{seller_id}' has not finished Stripe onboarding yet (charges not enabled)."
            )

        amount_cents = round(price_usd * 100)
        application_fee_cents = round(amount_cents * (MARKETPLACE_PLATFORM_FEE_PERCENT / 100.0))

        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            customer_email=buyer_email,
            line_items=[{
                "price_data": {
                    "currency": MARKETPLACE_CURRENCY,
                    "product_data": {"name": f"Voice license: {voice_name}"},
                    "unit_amount": amount_cents,
                },
                "quantity": 1,
            }],
            payment_intent_data={
                "application_fee_amount": application_fee_cents,
                "transfer_data": {"destination": seller_record["stripe_account_id"]},
            },
            metadata={
                "voice_id": voice_id,
                "seller_id": seller_id,
                "buyer_email": buyer_email or "",
                "purpose": purpose,
            },
            success_url=MARKETPLACE_CHECKOUT_SUCCESS_URL,
            cancel_url=MARKETPLACE_CHECKOUT_CANCEL_URL,
        )
        return {
            "session_id": session["id"],
            "checkout_url": session["url"],
            "amount_usd": amount_cents / 100.0,
            "platform_fee_usd": application_fee_cents / 100.0,
        }

    # ------------------------------------------------------------------
    # Webhook (server-side payment confirmation -- never trust the client redirect alone)
    # ------------------------------------------------------------------

    def construct_webhook_event(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """Verifies a Stripe webhook signature and returns the parsed event.
        Raises stripe.error.SignatureVerificationError on a bad/forged signature
        -- callers (the REST layer) should map that to an HTTP 400."""
        self._require_stripe()
        return stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)

    @staticmethod
    def extract_completed_checkout_metadata(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """If the event is a completed checkout payment, returns the license
        metadata needed to grant it (voice_id/seller_id/buyer_email/purpose plus
        the Stripe session/payment_intent ids for the audit trail). Returns None
        for any other event type so callers can safely ignore the rest of the
        Stripe webhook event catalog."""
        if event.get("type") != "checkout.session.completed":
            return None
        session = event["data"]["object"]
        if session.get("payment_status") != "paid":
            return None
        metadata = session.get("metadata", {}) or {}
        return {
            "voice_id": metadata.get("voice_id"),
            "seller_id": metadata.get("seller_id"),
            "buyer_email": metadata.get("buyer_email") or None,
            "purpose": metadata.get("purpose") or "",
            "stripe_session_id": session.get("id"),
            "stripe_payment_intent_id": session.get("payment_intent"),
            "amount_total_usd": (session.get("amount_total") or 0) / 100.0,
        }

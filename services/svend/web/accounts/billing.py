"""Stripe billing integration.

Handles:
- Checkout session creation
- Customer portal
- Webhook processing
- Subscription management
"""

import logging
from typing import Optional
from urllib.parse import urlencode

import stripe
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.auth.decorators import login_required

from core.encryption import hash_token

from .models import User, Subscription
from .constants import get_founder_availability

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Price ID to Tier mapping (includes legacy prices for existing subscribers)
PRICE_TO_TIER = {
    # Legacy prices (existing subscribers)
    "price_1SvoFZDQfJOZ4D24PhKMqiY5": User.Tier.FOUNDER,   # $19/month (legacy)
    "price_1SvoD0DQfJOZ4D24pIPmyitE": User.Tier.PRO,       # $29/month (legacy)
    "price_1SvoDiDQfJOZ4D24LYA2sCc5": User.Tier.TEAM,      # $79/month (legacy)
    "price_1SvoEGDQfJOZ4D24trvit3VM": User.Tier.ENTERPRISE, # $199/month (legacy)
    # New prices (Feb 2026)
    "price_1T0Y13DQfJOZ4D24GjaVOd09": User.Tier.PRO,       # $49/month
    "price_1T0Y36DQfJOZ4D24hhziBDe3": User.Tier.TEAM,      # $99/month
    "price_1T0Y42DQfJOZ4D245kTDgtal": User.Tier.ENTERPRISE, # $299/month
}

# Tier to Price ID mapping (for new checkouts)
TIER_TO_PRICE = {
    "founder": "price_1SvoFZDQfJOZ4D24PhKMqiY5",      # $19/month (first 50 slots)
    "pro": "price_1T0Y13DQfJOZ4D24GjaVOd09",          # $49/month
    "team": "price_1T0Y36DQfJOZ4D24hhziBDe3",         # $99/month
    "enterprise": "price_1T0Y42DQfJOZ4D245kTDgtal",    # $299/month
}

# Seat add-on removed — Team ($99) and Enterprise ($299) are per-seat prices.
SEAT_PRICE_ID = ""


# =============================================================================
# Helper Functions
# =============================================================================

def get_or_create_stripe_customer(user: User) -> str:
    """Get or create a Stripe customer for a user."""
    if user.stripe_customer_id:
        return user.stripe_customer_id

    customer = stripe.Customer.create(
        email=user.email,
        name=user.get_full_name() or user.username,
        metadata={"user_id": str(user.id)},
    )

    user.stripe_customer_id = customer.id
    user.stripe_customer_id_hash = hash_token(customer.id)
    user.save(update_fields=["stripe_customer_id", "stripe_customer_id_hash"])

    return customer.id


def add_org_seat(tenant) -> int:
    """Add a seat to the org.

    Each member needs their own Team/Enterprise subscription ($99/$299 per seat).
    This just tracks the local member cap.

    Returns the new max_members count.
    """
    tenant.max_members += 1
    tenant.save(update_fields=["max_members"])
    logger.info(f"Seat added for {tenant.name}: max_members={tenant.max_members}")
    return tenant.max_members


def remove_org_seat(tenant) -> int:
    """Remove a seat from the org.

    Returns the new max_members count.
    """
    tenant.max_members = max(1, tenant.max_members - 1)
    tenant.save(update_fields=["max_members"])
    logger.info(f"Seat removed for {tenant.name}: max_members={tenant.max_members}")
    return tenant.max_members


def sync_subscription_from_stripe(subscription_id: str) -> Optional[Subscription]:
    """Sync subscription data from Stripe."""
    try:
        stripe_sub = stripe.Subscription.retrieve(subscription_id)
    except stripe.error.StripeError as e:
        logger.error(f"Failed to retrieve subscription {subscription_id}: {e}")
        return None

    try:
        sub = Subscription.objects.get(stripe_subscription_id=subscription_id)
    except Subscription.DoesNotExist:
        # Find user by customer ID
        try:
            user = User.objects.get(stripe_customer_id_hash=hash_token(stripe_sub.customer))
        except User.DoesNotExist:
            logger.error(f"No user found for customer {stripe_sub.customer}")
            return None

        sub = Subscription.objects.create(
            user=user,
            stripe_subscription_id=subscription_id,
        )

    # Update subscription data
    from datetime import datetime
    from django.utils import timezone

    first_item = stripe_sub["items"]["data"][0] if stripe_sub["items"]["data"] else {}
    sub.stripe_price_id = first_item.get("price", {}).get("id", "") if first_item else ""
    sub.status = stripe_sub.status

    # Period fields moved to items in newer Stripe API versions
    period_start = (first_item.get("current_period_start")
                    or getattr(stripe_sub, "current_period_start", None))
    period_end = (first_item.get("current_period_end")
                  or getattr(stripe_sub, "current_period_end", None))
    sub.current_period_start = timezone.make_aware(
        datetime.fromtimestamp(period_start)
    ) if period_start else None
    sub.current_period_end = timezone.make_aware(
        datetime.fromtimestamp(period_end)
    ) if period_end else None
    sub.cancel_at_period_end = stripe_sub.cancel_at_period_end
    sub.save()

    # Update user tier based on subscription status and price
    user = sub.user
    if sub.is_active:
        # Determine tier from price ID
        tier = PRICE_TO_TIER.get(sub.stripe_price_id)
        if tier is None:
            logger.warning(f"Unknown Stripe price ID: {sub.stripe_price_id} — defaulting to FREE")
            tier = User.Tier.FREE
        user.tier = tier
        user.subscription_active = True
        user.subscription_ends_at = sub.current_period_end
    else:
        user.tier = User.Tier.FREE
        user.subscription_active = False
    user.save(update_fields=["tier", "subscription_active", "subscription_ends_at"])

    return sub


# =============================================================================
# Views
# =============================================================================

@login_required
def create_checkout_session(request: HttpRequest):
    """Create a Stripe Checkout session for subscription.

    Accepts ?plan= query param: founder, pro, team, enterprise
    Defaults to pro if not specified. Founder limited to first 50 slots.
    """
    if not settings.STRIPE_SECRET_KEY:
        return JsonResponse({"error": "Billing not configured"}, status=503)

    user = request.user

    # Check if already subscribed
    if hasattr(user, "subscription") and user.subscription.is_active:
        return redirect("/app/settings/?error=already_subscribed")

    # Get plan from query param
    plan = request.GET.get("plan", "pro").lower()
    price_id = TIER_TO_PRICE.get(plan)

    if not price_id:
        return redirect("/app/settings/?error=invalid_plan")

    # Founder tier — check slot availability
    if plan == "founder":
        availability = get_founder_availability()
        if not availability["available"]:
            return redirect("/app/settings/?error=founder_sold_out")

    try:
        customer_id = get_or_create_stripe_customer(user)

        # Build URLs
        # NOTE: {CHECKOUT_SESSION_ID} must NOT be URL-encoded — Stripe replaces it.
        # build_absolute_uri encodes braces, so construct manually.
        base = request.build_absolute_uri("/billing/success/")
        success_url = base + "?session_id={CHECKOUT_SESSION_ID}"
        cancel_url = request.build_absolute_uri("/app/settings/?checkout=cancelled")

        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[
                {
                    "price": price_id,
                    "quantity": 1,
                }
            ],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"user_id": str(user.id), "plan": plan},
        )

        # Redirect directly to Stripe checkout
        return redirect(session.url)

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout: {e}")
        return redirect("/app/settings/?error=checkout_failed")


@login_required
def create_portal_session(request: HttpRequest):
    """Create a Stripe Customer Portal session for subscription management."""
    if not settings.STRIPE_SECRET_KEY:
        return redirect("/app/settings/?error=billing_not_configured")

    user = request.user

    if not user.stripe_customer_id:
        return redirect("/app/settings/?error=no_billing_account")

    try:
        return_url = request.build_absolute_uri("/app/settings/")

        session = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=return_url,
        )

        # Redirect directly to Stripe portal
        return redirect(session.url)

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating portal: {e}")
        return redirect("/app/settings/?error=portal_failed")


@login_required
def checkout_success(request: HttpRequest) -> HttpResponse:
    """Handle successful checkout redirect."""
    session_id = request.GET.get("session_id")

    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)

            # Verify session belongs to logged-in user
            if session.customer != request.user.stripe_customer_id:
                logger.warning(
                    f"Checkout session customer mismatch: session={session.customer} "
                    f"user={request.user.stripe_customer_id}"
                )
                return redirect("/app/settings/?error=session_mismatch")

            if session.subscription:
                sync_subscription_from_stripe(session.subscription)
        except stripe.error.StripeError as e:
            logger.error(f"Error retrieving checkout session: {e}")

    return redirect("/app/?upgraded=true")


@login_required
def checkout_cancel(request: HttpRequest) -> HttpResponse:
    """Handle cancelled checkout."""
    return redirect("/app/settings/?checkout=cancelled")


@csrf_exempt
@require_POST
def stripe_webhook(request: HttpRequest) -> HttpResponse:
    """Handle Stripe webhook events."""
    payload = request.body
    sig_header = request.META.get("HTTP_STRIPE_SIGNATURE", "")

    if not settings.STRIPE_WEBHOOK_SECRET:
        logger.warning("Stripe webhook secret not configured")
        return HttpResponse(status=400)

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        logger.error("Invalid webhook payload")
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid webhook signature")
        return HttpResponse(status=400)

    # Handle events
    event_type = event["type"]
    data = event["data"]["object"]

    logger.info(f"Stripe webhook: {event_type}")

    if event_type == "checkout.session.completed":
        # New subscription created
        if data.get("subscription"):
            sync_subscription_from_stripe(data["subscription"])

    elif event_type == "customer.subscription.updated":
        sync_subscription_from_stripe(data["id"])

    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled
        try:
            sub = Subscription.objects.get(stripe_subscription_id=data["id"])
            sub.status = Subscription.Status.CANCELED
            sub.save(update_fields=["status"])

            # Downgrade user
            user = sub.user
            user.tier = User.Tier.FREE
            user.subscription_active = False
            user.subscription_ends_at = None
            user.save(update_fields=["tier", "subscription_active", "subscription_ends_at"])

        except Subscription.DoesNotExist:
            logger.warning(f"Subscription not found: {data['id']}")

    elif event_type == "invoice.payment_failed":
        # Payment failed — mark past due AND downgrade tier
        subscription_id = data.get("subscription")
        if subscription_id:
            try:
                sub = Subscription.objects.get(stripe_subscription_id=subscription_id)
                sub.status = Subscription.Status.PAST_DUE
                sub.save(update_fields=["status"])

                # Downgrade user to prevent free access on failed payment
                user = sub.user
                user.tier = User.Tier.FREE
                user.subscription_active = False
                user.save(update_fields=["tier", "subscription_active"])
                logger.info(f"User {user.username} downgraded due to payment failure")
            except Subscription.DoesNotExist:
                pass

    elif event_type == "invoice.paid":
        # Payment succeeded (renewal)
        subscription_id = data.get("subscription")
        if subscription_id:
            sync_subscription_from_stripe(subscription_id)

    return HttpResponse(status=200)


@login_required
def subscription_status(request: HttpRequest) -> JsonResponse:
    """Get current subscription status."""
    user = request.user

    data = {
        "tier": user.tier,
        "daily_limit": user.daily_limit,
        "queries_today": user.queries_today,
        "queries_remaining": max(0, user.daily_limit - user.queries_today),
    }

    if hasattr(user, "subscription"):
        sub = user.subscription
        data["subscription"] = {
            "status": sub.status,
            "is_active": sub.is_active,
            "current_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
            "cancel_at_period_end": sub.cancel_at_period_end,
        }

    return JsonResponse(data)


@require_http_methods(["GET"])
def founder_availability(request: HttpRequest) -> JsonResponse:
    """Check if founder pricing is still available.

    Public endpoint (no auth required) for landing page to check.
    """
    from .constants import get_founder_availability

    return JsonResponse(get_founder_availability())

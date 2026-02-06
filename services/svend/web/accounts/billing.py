"""Stripe billing integration.

Handles:
- Checkout session creation
- Customer portal
- Webhook processing
- Subscription management
"""

import logging
from typing import Optional

import stripe
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.auth.decorators import login_required

from .models import User, Subscription

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Price ID to Tier mapping
PRICE_TO_TIER = {
    "price_1SvoFZDQfJOZ4D24PhKMqiY5": User.Tier.FOUNDER,   # $19/month
    "price_1SvoD0DQfJOZ4D24pIPmyitE": User.Tier.PRO,       # $29/month
    "price_1SvoDiDQfJOZ4D24LYA2sCc5": User.Tier.TEAM,      # $79/month
    "price_1SvoEGDQfJOZ4D24trvit3VM": User.Tier.ENTERPRISE, # $199/month
}

# Tier to Price ID mapping (for checkout)
TIER_TO_PRICE = {
    "founder": "price_1SvoFZDQfJOZ4D24PhKMqiY5",
    "pro": "price_1SvoD0DQfJOZ4D24pIPmyitE",
    "team": "price_1SvoDiDQfJOZ4D24LYA2sCc5",
    "enterprise": "price_1SvoEGDQfJOZ4D24trvit3VM",
}


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
    user.save(update_fields=["stripe_customer_id"])

    return customer.id


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
            user = User.objects.get(stripe_customer_id=stripe_sub.customer)
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

    sub.stripe_price_id = stripe_sub["items"]["data"][0]["price"]["id"] if stripe_sub["items"]["data"] else ""
    sub.status = stripe_sub.status
    sub.current_period_start = timezone.make_aware(
        datetime.fromtimestamp(stripe_sub.current_period_start)
    ) if stripe_sub.current_period_start else None
    sub.current_period_end = timezone.make_aware(
        datetime.fromtimestamp(stripe_sub.current_period_end)
    ) if stripe_sub.current_period_end else None
    sub.cancel_at_period_end = stripe_sub.cancel_at_period_end
    sub.save()

    # Update user tier based on subscription status and price
    user = sub.user
    if sub.is_active:
        # Determine tier from price ID
        tier = PRICE_TO_TIER.get(sub.stripe_price_id, User.Tier.PRO)
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
    Defaults to pro if not specified.
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

    try:
        customer_id = get_or_create_stripe_customer(user)

        # Build URLs
        success_url = request.build_absolute_uri("/billing/success?session_id={CHECKOUT_SESSION_ID}")
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
        return redirect(f"/app/settings/?error={str(e)[:50]}")


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
        return redirect(f"/app/settings/?error={str(e)[:50]}")


@login_required
def checkout_success(request: HttpRequest) -> HttpResponse:
    """Handle successful checkout redirect."""
    session_id = request.GET.get("session_id")

    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            if session.subscription:
                sync_subscription_from_stripe(session.subscription)
        except stripe.error.StripeError as e:
            logger.error(f"Error retrieving checkout session: {e}")

    return redirect("/?upgraded=true")


@login_required
def checkout_cancel(request: HttpRequest) -> HttpResponse:
    """Handle cancelled checkout."""
    return redirect("/?checkout=cancelled")


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
            user.save(update_fields=["tier", "subscription_active"])

        except Subscription.DoesNotExist:
            logger.warning(f"Subscription not found: {data['id']}")

    elif event_type == "invoice.payment_failed":
        # Payment failed
        subscription_id = data.get("subscription")
        if subscription_id:
            try:
                sub = Subscription.objects.get(stripe_subscription_id=subscription_id)
                sub.status = Subscription.Status.PAST_DUE
                sub.save(update_fields=["status"])
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

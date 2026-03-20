"""Stripe billing integration.

Handles:
- Checkout session creation
- Customer portal
- Webhook processing
- Subscription management
"""

import logging

import stripe
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST

from core.encryption import hash_token

from .constants import get_founder_availability
from .models import Subscription, User

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Price ID to Tier mapping (includes legacy + all regional prices)
PRICE_TO_TIER = {
    # Legacy prices (existing subscribers)
    "price_1SvoFZDQfJOZ4D24PhKMqiY5": User.Tier.FOUNDER,  # $19/month (legacy)
    "price_1SvoD0DQfJOZ4D24pIPmyitE": User.Tier.PRO,  # $29/month (legacy)
    "price_1SvoDiDQfJOZ4D24LYA2sCc5": User.Tier.TEAM,  # $79/month (legacy)
    "price_1SvoEGDQfJOZ4D24trvit3VM": User.Tier.ENTERPRISE,  # $199/month (legacy)
    # USD (Feb 2026)
    "price_1T0Y13DQfJOZ4D24GjaVOd09": User.Tier.PRO,  # $49/month
    "price_1T0Y36DQfJOZ4D24hhziBDe3": User.Tier.TEAM,  # $99/month
    "price_1T0Y42DQfJOZ4D245kTDgtal": User.Tier.ENTERPRISE,  # $299/month
    # INR — India
    "price_1T17YfDQfJOZ4D24dmfpjXIx": User.Tier.PRO,  # ₹1,499/month
    "price_1T17Z8DQfJOZ4D24ZyHZrCk5": User.Tier.TEAM,  # ₹3,499/month
    "price_1T17ZdDQfJOZ4D24YfN6KjJN": User.Tier.ENTERPRISE,  # ₹9,999/month
    "price_1T1EUtDQfJOZ4D24YXFV4Nzy": User.Tier.PRO,  # ₹749/month (student)
    # VND — Vietnam
    "price_1T17iEDQfJOZ4D24cJxOjmJC": User.Tier.PRO,  # 349,000 VND/month
    "price_1T17k0DQfJOZ4D24pTu9aSel": User.Tier.TEAM,  # 799,000 VND/month
    "price_1T17kYDQfJOZ4D248bVyQZZd": User.Tier.ENTERPRISE,  # 2,499,000 VND/month
    # UAH — Ukraine
    "price_1T17r8DQfJOZ4D24EnYXIEdy": User.Tier.PRO,  # 349 UAH/month
    "price_1T17rlDQfJOZ4D24EflhMPmz": User.Tier.TEAM,  # 899 UAH/month
    "price_1T17sFDQfJOZ4D24uHsYWFJp": User.Tier.ENTERPRISE,  # 2,999 UAH/month
    # PHP — Philippines
    "price_1T17t1DQfJOZ4D24MVi06BLK": User.Tier.PRO,  # 1,290 PHP/month
    "price_1T17tZDQfJOZ4D24BMLe61qa": User.Tier.TEAM,  # 2,990 PHP/month
    "price_1T17tyDQfJOZ4D24Ct4LBNuh": User.Tier.ENTERPRISE,  # 8,990 PHP/month
    # MYR — Malaysia
    "price_1T17unDQfJOZ4D24eSHwoZzO": User.Tier.PRO,  # 99 MYR/month
    "price_1T17vbDQfJOZ4D24RWobYBJc": User.Tier.TEAM,  # 229 MYR/month
    "price_1T17w5DQfJOZ4D24Kp6X2IFc": User.Tier.ENTERPRISE,  # 699 MYR/month
    # IDR — Indonesia
    "price_1T17y8DQfJOZ4D248RnlqMgq": User.Tier.PRO,  # 249,000 IDR/month
    "price_1T17zSDQfJOZ4D24kOE1hyPG": User.Tier.TEAM,  # 579,000 IDR/month
    "price_1T181ODQfJOZ4D24gm3ogLAw": User.Tier.ENTERPRISE,  # 1,799,000 IDR/month
    # MXN — Mexico
    "price_1T18nZDQfJOZ4D24Qj0lumUM": User.Tier.PRO,  # 449 MXN/month
    "price_1T18o6DQfJOZ4D24MJAKV3Cm": User.Tier.TEAM,  # 899 MXN/month
    "price_1T18oYDQfJOZ4D24VCqzU7WH": User.Tier.ENTERPRISE,  # 2,490 MXN/month
    # AED — GCC / Middle East
    "price_1T44slDQfJOZ4D24muyszMUV": User.Tier.PRO,  # 149 AED/month
    "price_1T44vSDQfJOZ4D24bVk265Bd": User.Tier.TEAM,  # 349 AED/month
    "price_1T44y3DQfJOZ4D24OKbqnHBn": User.Tier.ENTERPRISE,  # 999 AED/month
    # ZAR — South Africa
    "price_1T44svDQfJOZ4D24McEo6CeA": User.Tier.PRO,  # 201 ZAR/month
    "price_1T44viDQfJOZ4D24WlVGsr3r": User.Tier.TEAM,  # 419 ZAR/month
    "price_1T44yBDQfJOZ4D24ozZt0Ys1": User.Tier.ENTERPRISE,  # 1,256 ZAR/month
    "price_1TC07QDQfJOZ4D248eS6wUBt": User.Tier.PRO,  # 101 ZAR/month (student/alumni 50%)
    "price_1TC09zDQfJOZ4D24rpPEzQ3w": User.Tier.PRO,  # 84 ZAR/month (NGO)
    "price_1TC0B8DQfJOZ4D247rpf18du": User.Tier.TEAM,  # 251 ZAR/month (NGO)
    # KES — Kenya
    "price_1T44tMDQfJOZ4D24gGM77q3d": User.Tier.PRO,  # 1,990 KES/month
    "price_1T44w2DQfJOZ4D242THxiHz5": User.Tier.TEAM,  # 4,490 KES/month
    "price_1T44yVDQfJOZ4D248p1K9hjH": User.Tier.ENTERPRISE,  # 13,990 KES/month
    # NGN — Nigeria
    "price_1T44tqDQfJOZ4D24BtkJCHPM": User.Tier.PRO,  # 4,990 NGN/month
    "price_1T44wFDQfJOZ4D24CK2fPnmj": User.Tier.TEAM,  # 11,990 NGN/month
    "price_1T44ykDQfJOZ4D24T82bsMPw": User.Tier.ENTERPRISE,  # 34,990 NGN/month
    # BRL — Brazil
    "price_1T44u4DQfJOZ4D246jcdqY5D": User.Tier.PRO,  # 99 BRL/month
    "price_1T44wUDQfJOZ4D244MQMLTJD": User.Tier.TEAM,  # 229 BRL/month
    "price_1T44yuDQfJOZ4D246o3i1BD5": User.Tier.ENTERPRISE,  # 699 BRL/month
    # COP — Colombia
    "price_1T44uhDQfJOZ4D24zr3vSIjg": User.Tier.PRO,  # 59,900 COP/month
    "price_1T44wjDQfJOZ4D24gKJoo2mq": User.Tier.TEAM,  # 139,900 COP/month
    "price_1T44zADQfJOZ4D24nB2Xfr9Y": User.Tier.ENTERPRISE,  # 449,900 COP/month
    # THB — Thailand
    "price_1T44v0DQfJOZ4D24mjX9HweE": User.Tier.PRO,  # 749 THB/month
    "price_1T44wwDQfJOZ4D24K1NzrX7n": User.Tier.TEAM,  # 1,690 THB/month
    "price_1T44zLDQfJOZ4D24yQVXbRpj": User.Tier.ENTERPRISE,  # 4,990 THB/month
}

# Regional price tables — keyed by region code
# Each region maps tier → Stripe price ID
REGIONAL_PRICES = {
    "us": {
        "founder": "price_1SvoFZDQfJOZ4D24PhKMqiY5",  # $19/month (first 50 slots)
        "pro": "price_1T0Y13DQfJOZ4D24GjaVOd09",  # $49/month
        "team": "price_1T0Y36DQfJOZ4D24hhziBDe3",  # $99/month
        "enterprise": "price_1T0Y42DQfJOZ4D245kTDgtal",  # $299/month
    },
    "in": {  # India
        "pro": "price_1T17YfDQfJOZ4D24dmfpjXIx",  # ₹1,499/month
        "team": "price_1T17Z8DQfJOZ4D24ZyHZrCk5",  # ₹3,499/month
        "enterprise": "price_1T17ZdDQfJOZ4D24YfN6KjJN",  # ₹9,999/month
    },
    "vn": {  # Vietnam
        "pro": "price_1T17iEDQfJOZ4D24cJxOjmJC",  # 349,000 VND/month
        "team": "price_1T17k0DQfJOZ4D24pTu9aSel",  # 799,000 VND/month
        "enterprise": "price_1T17kYDQfJOZ4D248bVyQZZd",  # 2,499,000 VND/month
    },
    "ua": {  # Ukraine
        "pro": "price_1T17r8DQfJOZ4D24EnYXIEdy",  # 349 UAH/month
        "team": "price_1T17rlDQfJOZ4D24EflhMPmz",  # 899 UAH/month
        "enterprise": "price_1T17sFDQfJOZ4D24uHsYWFJp",  # 2,999 UAH/month
    },
    "ph": {  # Philippines
        "pro": "price_1T17t1DQfJOZ4D24MVi06BLK",  # 1,290 PHP/month
        "team": "price_1T17tZDQfJOZ4D24BMLe61qa",  # 2,990 PHP/month
        "enterprise": "price_1T17tyDQfJOZ4D24Ct4LBNuh",  # 8,990 PHP/month
    },
    "my": {  # Malaysia
        "pro": "price_1T17unDQfJOZ4D24eSHwoZzO",  # 99 MYR/month
        "team": "price_1T17vbDQfJOZ4D24RWobYBJc",  # 229 MYR/month
        "enterprise": "price_1T17w5DQfJOZ4D24Kp6X2IFc",  # 699 MYR/month
    },
    "id": {  # Indonesia
        "pro": "price_1T17y8DQfJOZ4D248RnlqMgq",  # 249,000 IDR/month
        "team": "price_1T17zSDQfJOZ4D24kOE1hyPG",  # 579,000 IDR/month
        "enterprise": "price_1T181ODQfJOZ4D24gm3ogLAw",  # 1,799,000 IDR/month
    },
    "mx": {  # Mexico
        "pro": "price_1T18nZDQfJOZ4D24Qj0lumUM",  # 449 MXN/month
        "team": "price_1T18o6DQfJOZ4D24MJAKV3Cm",  # 899 MXN/month
        "enterprise": "price_1T18oYDQfJOZ4D24VCqzU7WH",  # 2,490 MXN/month
    },
    "ae": {  # GCC / Middle East (AED)
        "pro": "price_1T44slDQfJOZ4D24muyszMUV",  # 149 AED/month
        "team": "price_1T44vSDQfJOZ4D24bVk265Bd",  # 349 AED/month
        "enterprise": "price_1T44y3DQfJOZ4D24OKbqnHBn",  # 999 AED/month
    },
    "za": {  # South Africa (ZAR)
        "pro": "price_1T44svDQfJOZ4D24McEo6CeA",  # 201 ZAR/month
        "team": "price_1T44viDQfJOZ4D24WlVGsr3r",  # 419 ZAR/month
        "enterprise": "price_1T44yBDQfJOZ4D24ozZt0Ys1",  # 1,256 ZAR/month
        "student_pro": "price_1TC07QDQfJOZ4D248eS6wUBt",  # 101 ZAR/month (50% alumni)
        "ngo_pro": "price_1TC09zDQfJOZ4D24rpPEzQ3w",  # 84 ZAR/month (NGO)
        "ngo_team": "price_1TC0B8DQfJOZ4D247rpf18du",  # 251 ZAR/month (NGO)
    },
    "ke": {  # Kenya (KES)
        "pro": "price_1T44tMDQfJOZ4D24gGM77q3d",  # 1,990 KES/month
        "team": "price_1T44w2DQfJOZ4D242THxiHz5",  # 4,490 KES/month
        "enterprise": "price_1T44yVDQfJOZ4D248p1K9hjH",  # 13,990 KES/month
    },
    "ng": {  # Nigeria (NGN)
        "pro": "price_1T44tqDQfJOZ4D24BtkJCHPM",  # 4,990 NGN/month
        "team": "price_1T44wFDQfJOZ4D24CK2fPnmj",  # 11,990 NGN/month
        "enterprise": "price_1T44ykDQfJOZ4D24T82bsMPw",  # 34,990 NGN/month
    },
    "br": {  # Brazil (BRL)
        "pro": "price_1T44u4DQfJOZ4D246jcdqY5D",  # 99 BRL/month
        "team": "price_1T44wUDQfJOZ4D244MQMLTJD",  # 229 BRL/month
        "enterprise": "price_1T44yuDQfJOZ4D246o3i1BD5",  # 699 BRL/month
    },
    "co": {  # Colombia (COP)
        "pro": "price_1T44uhDQfJOZ4D24zr3vSIjg",  # 59,900 COP/month
        "team": "price_1T44wjDQfJOZ4D24gKJoo2mq",  # 139,900 COP/month
        "enterprise": "price_1T44zADQfJOZ4D24nB2Xfr9Y",  # 449,900 COP/month
    },
    "th": {  # Thailand (THB)
        "pro": "price_1T44v0DQfJOZ4D24mjX9HweE",  # 749 THB/month
        "team": "price_1T44wwDQfJOZ4D24K1NzrX7n",  # 1,690 THB/month
        "enterprise": "price_1T44zLDQfJOZ4D24yQVXbRpj",  # 4,990 THB/month
    },
}

# Country code → region mapping (for countries that share a regional price)
COUNTRY_TO_REGION = {
    # Direct matches
    "IN": "in",
    "VN": "vn",
    "UA": "ua",
    "PH": "ph",
    "MY": "my",
    "ID": "id",
    "MX": "mx",
    # GCC / Middle East
    "AE": "ae",
    "SA": "ae",
    "QA": "ae",
    "KW": "ae",
    "BH": "ae",
    "OM": "ae",
    # Africa
    "ZA": "za",
    "BW": "za",  # South Africa + Botswana (ZAR region)
    "KE": "ke",  # Kenya
    "NG": "ng",  # Nigeria
    # Latin America
    "BR": "br",
    "CO": "co",
    # Southeast Asia
    "TH": "th",
    # US/EU/AU — default USD pricing
    "US": "us",
    "CA": "us",
    "GB": "us",
    "AU": "us",
    "DE": "us",
    "FR": "us",
    "NL": "us",
    "SE": "us",
}

# Default tier-to-price (USD fallback for new checkouts)
TIER_TO_PRICE = {
    "founder": "price_1SvoFZDQfJOZ4D24PhKMqiY5",  # $19/month (first 50 slots)
    "pro": "price_1T0Y13DQfJOZ4D24GjaVOd09",  # $49/month
    "team": "price_1T0Y36DQfJOZ4D24hhziBDe3",  # $99/month
    "enterprise": "price_1T0Y42DQfJOZ4D245kTDgtal",  # $299/month
}


def get_price_for_region(plan: str, region: str = "") -> str:
    """Get the Stripe price ID for a plan+region combo.

    Falls back to USD if region not found or plan not available regionally.
    """
    if region and region in REGIONAL_PRICES:
        regional = REGIONAL_PRICES[region]
        if plan in regional:
            return regional[plan]
    # Fallback to USD
    return TIER_TO_PRICE.get(plan, "")


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


def sync_subscription_from_stripe(subscription_id: str) -> Subscription | None:
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

        # User may already have a subscription from a previous plan — update it
        # rather than creating a duplicate (subscriptions_user_id_key is unique).
        sub, _created = Subscription.objects.update_or_create(
            user=user,
            defaults={"stripe_subscription_id": subscription_id},
        )

    # Update subscription data
    from datetime import datetime

    from django.utils import timezone

    first_item = stripe_sub["items"]["data"][0] if stripe_sub["items"]["data"] else {}
    sub.stripe_price_id = first_item.get("price", {}).get("id", "") if first_item else ""
    sub.status = stripe_sub.status

    # Period fields moved to items in newer Stripe API versions
    period_start = first_item.get("current_period_start") or getattr(stripe_sub, "current_period_start", None)
    period_end = first_item.get("current_period_end") or getattr(stripe_sub, "current_period_end", None)
    sub.current_period_start = timezone.make_aware(datetime.fromtimestamp(period_start)) if period_start else None
    sub.current_period_end = timezone.make_aware(datetime.fromtimestamp(period_end)) if period_end else None
    sub.is_cancel_at_period_end = getattr(stripe_sub, "cancel_at_period_end", False)
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
        user.is_subscription_active = True
        user.subscription_ends_at = sub.current_period_end
    else:
        user.tier = User.Tier.FREE
        user.is_subscription_active = False
    user.save(update_fields=["tier", "is_subscription_active", "subscription_ends_at"])

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

    # Get plan and region from query params
    plan = request.GET.get("plan", "pro").lower()
    region = request.GET.get("region", "").lower()

    # Map country code to region if a full country code was passed
    if len(region) == 2 and region.upper() in COUNTRY_TO_REGION:
        region = COUNTRY_TO_REGION[region.upper()]

    # Fallback: detect region from Cloudflare header if param is missing
    if not region:
        country = request.META.get("HTTP_CF_IPCOUNTRY", "").upper()
        if country and country not in ("XX", "T1"):
            region = COUNTRY_TO_REGION.get(country, "")

    price_id = get_price_for_region(plan, region)

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

        # 14-day free trial for Team and Enterprise (QMS modules)
        TRIAL_PLANS = {"team", "enterprise"}
        checkout_kwargs = {
            "customer": customer_id,
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": {"user_id": str(user.id), "plan": plan},
        }
        if plan in TRIAL_PLANS:
            checkout_kwargs["subscription_data"] = {"trial_period_days": 14}

        # Create checkout session
        session = stripe.checkout.Session.create(**checkout_kwargs)

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
        event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
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
            user.is_subscription_active = False
            user.subscription_ends_at = None
            user.save(update_fields=["tier", "is_subscription_active", "subscription_ends_at"])

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
                user.is_subscription_active = False
                user.save(update_fields=["tier", "is_subscription_active"])
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
            "cancel_at_period_end": sub.is_cancel_at_period_end,
        }

    return JsonResponse(data)


@require_http_methods(["GET"])
def founder_availability(request: HttpRequest) -> JsonResponse:
    """Check if founder pricing is still available.

    Public endpoint (no auth required) for landing page to check.
    """
    from .constants import get_founder_availability

    return JsonResponse(get_founder_availability())

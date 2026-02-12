"""Account and billing URLs."""

from django.urls import path

from . import billing

app_name = "accounts"

urlpatterns = [
    # Billing
    path("billing/checkout/", billing.create_checkout_session, name="checkout"),
    path("billing/portal/", billing.create_portal_session, name="portal"),
    path("billing/success/", billing.checkout_success, name="checkout_success"),
    path("billing/cancel/", billing.checkout_cancel, name="checkout_cancel"),
    path("billing/status/", billing.subscription_status, name="subscription_status"),
    path("billing/founder-availability/", billing.founder_availability, name="founder_availability"),

    # Stripe webhook (no auth required, verified by signature)
    path("webhooks/stripe/", billing.stripe_webhook, name="stripe_webhook"),
    path("billing/webhook/", billing.stripe_webhook, name="stripe_webhook_alias"),
]

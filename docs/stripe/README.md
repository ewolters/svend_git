# Stripe — Payment Processing

**Purpose:** Subscription billing, checkout, and customer management.
**Dashboard:** https://dashboard.stripe.com
**Compliance:** PCI DSS Level 1, SOC 2 Type II

---

## How It Works

```
User clicks "Start Trial" → Stripe Checkout (hosted) → Webhook → Subscription activated
```

Svend never touches card data. Stripe.js handles payment client-side, Stripe Checkout hosts the payment form, and webhooks notify us of lifecycle events.

## Credentials

| Secret | Location | Purpose |
|--------|----------|---------|
| `SVEND_STRIPE_SECRET_KEY` | `~/.svend_env` / `svend_config/config.py` | Server-side API calls |
| `SVEND_STRIPE_WEBHOOK_SECRET` | `~/.svend_env` / `svend_config/config.py` | Webhook signature verification |
| `SVEND_STRIPE_PRICE_ID_PRO` | `svend_config/config.py` | USD Pro plan price ID |
| Regional price IDs | `accounts/billing.py` | 19+ price IDs across 15 currencies |

**Never commit these to git.** They're loaded from env vars or the config module.

## Integration Points

| File | Purpose |
|------|---------|
| `accounts/billing.py` | Core billing logic: checkout, portal, webhooks, regional pricing |
| `accounts/models.py` | `Subscription` model (tier, stripe_customer_id, stripe_subscription_id) |
| `accounts/views.py` | Billing endpoints |
| `templates/register.html` | Client-side Stripe.js integration |

## Webhook Events We Handle

| Event | Action |
|-------|--------|
| `checkout.session.completed` | Activate subscription, set tier |
| `customer.subscription.updated` | Sync tier changes |
| `customer.subscription.deleted` | Downgrade to free |
| `invoice.payment_failed` | Flag account |
| `invoice.paid` | Clear payment failure flags |

Webhook endpoint: `/webhooks/stripe/`

## Pricing Tiers

| Tier | USD | Trial |
|------|-----|-------|
| Free | $0 | N/A |
| Founder (legacy) | $19/mo | No |
| Pro | $49/mo | No |
| Team | $99/mo | 14-day |
| Enterprise | $299/mo | 14-day |

15 regional currencies supported (INR, VND, UAH, PHP, MYR, IDR, MXN, AED, ZAR, KES, NGN, BRL, COP, THB). Regional pricing detected via Cloudflare `CF-IPCountry` header.

## Common Tasks

### Check a customer's subscription in Stripe
```bash
# From Django shell
python3 manage.py shell
>>> from accounts.models import User
>>> u = User.objects.get(email="customer@example.com")
>>> u.subscription.stripe_customer_id  # Look this up in Stripe dashboard
```

### Test webhooks locally
```bash
stripe listen --forward-to localhost:8000/webhooks/stripe/
```

### View webhook logs
Stripe Dashboard → Developers → Webhooks → select endpoint → Recent deliveries

## Known Gaps (SOC 2)

- **PAR-07:** No webhook event deduplication (replay possible) — tracked as REM-04

**BILL-001: BILLING & SUBSCRIPTION STANDARD**

**Version:** 1.0
**Status:** APPROVED
**Date:** 2026-03-03
**Supersedes:** None
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 ≥ 1.0 (Documentation Structure — §4 Section Structure, §5 Metadata)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax — §4)
- SEC-001 ≥ 1.0 (Security Architecture — §5 Encryption, §7 Webhook Verification)
- DAT-001 ≥ 1.0 (Data Model — §4 UUID PKs, §5 Field Patterns)
- SOC 2 CC6.1 (Logical Access Controls — feature gating)
- SOC 2 CC6.7 (Restriction of Access to System Configurations)
**Related Standards:**
- API-001 ≥ 1.0 (API Design — billing endpoints)
- CHG-001 ≥ 1.0 (Change Management — pricing changes)
- ERR-001 ≥ 1.0 (Error Handling — payment failures)

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

BILL-001 defines the subscription tier system, Stripe integration patterns, regional pricing strategy, feature gating, and payment security requirements for the Svend billing system.

**Core Principle:**

> Users pay for value, not access. Feature gating is a business contract enforced in code.
> Every payment event is verified, every tier change is logged, and no user gets free access through code shortcuts.

### **1.2 Scope**

**Applies to:**
- All files in `services/svend/web/accounts/`
- Billing endpoints in `accounts/urls.py`
- Permission decorators in `accounts/permissions.py`
- Stripe webhook handler in `accounts/billing.py`
- Feature gating in `accounts/constants.py`

**Does NOT apply to:**
- Internal staff tools (no billing required)
- Public marketing pages
- Email tracking endpoints

---

## **2. NORMATIVE REFERENCES**

### **2.1 Kjerne Standards**

| Standard | Section | Requirement |
|----------|---------|-------------|
| **SEC-001** | §5 | Encryption at rest for payment data |
| **SEC-001** | §7 | Webhook signature verification |
| **DAT-001** | §4 | UUID primary keys |
| **API-001** | §5 | Error envelope for billing responses |

### **2.2 External Standards**

| Standard | Clause | Requirement |
|----------|--------|-------------|
| **PCI DSS** | 3.4 | Render PAN unreadable (delegated to Stripe) |
| **SOC 2** | CC6.1 | Logical access controls |
| **SOC 2** | CC6.7 | Restriction of access to configurations |
| **GDPR** | Art. 17 | Right to deletion (account + billing data) |

---

## **3. TERMINOLOGY**

| Term | Definition |
|------|------------|
| **Tier** | Subscription level: FREE, FOUNDER, PRO, TEAM, ENTERPRISE. Determines feature access and daily limits. |
| **Feature gate** | Code-level check that restricts functionality based on user tier. Enforced via decorators. |
| **Regional pricing** | Localized pricing in 14+ currencies based on Cloudflare CF-IPCountry header. |
| **Founder lock** | Legacy pricing ($19/mo) preserved for early adopters. `is_founder_locked=True` prevents tier reassignment. |
| **Daily limit** | Maximum API queries per day per tier. Reset at `queries_reset_at`. |
| **Webhook** | Stripe-to-Svend HTTP callback for subscription events. HMAC-SHA256 verified. |

---

## **4. SUBSCRIPTION TIERS**

### **4.1 Tier Matrix**

<!-- assert: Five subscription tiers with defined feature access and daily limits | check=bill-tiers -->
<!-- impl: accounts/constants.py:TIER_LIMITS -->
<!-- impl: accounts/models.py:User -->
<!-- test: api.tests.TierFeatureTest.test_tier_limits_correct -->
<!-- test: api.tests.RegistrationTest.test_register_new_user_is_free_tier -->
<!-- test: accounts.tests.TierConstantsTest.test_all_five_tiers_defined -->
<!-- test: accounts.tests.TierConstantsTest.test_tier_limits_has_entry_for_each_tier -->
<!-- test: accounts.tests.TierConstantsTest.test_get_daily_limit_known_tiers -->
<!-- test: accounts.tests.TierConstantsTest.test_get_daily_limit_unknown_tier_falls_back_to_free -->
<!-- test: accounts.tests.UserModelTest.test_daily_limit_matches_tier_limits -->

<!-- table: field-definitions -->
| Tier | Price | Queries/Day | Key Features |
|------|-------|-------------|--------------|
| FREE | $0 | 5 | Basic DSW only |
| FOUNDER | $19/mo | 50 | Legacy, rate-locked, no longer sold |
| PRO | $49/mo | 50 | Full tools, basic ML, Forge API |
| TEAM | $99/mo | 200 | Collaboration, priority support |
| ENTERPRISE | $299/mo | 1000 | AI Assistant, Hoshin Kanri CI, SSO |

### **4.2 Feature Gating Matrix**

<!-- assert: Feature access controlled by TIER_FEATURES constant | check=bill-features -->
<!-- impl: accounts/constants.py:TIER_FEATURES -->
<!-- test: api.tests.TierFeatureTest.test_free_has_no_collaboration -->
<!-- test: api.tests.TierFeatureTest.test_free_has_no_full_tools -->
<!-- test: api.tests.TierFeatureTest.test_pro_has_full_tools -->
<!-- test: api.tests.TierFeatureTest.test_pro_has_no_collaboration -->
<!-- test: api.tests.TierFeatureTest.test_team_has_collaboration -->
<!-- test: api.tests.TierFeatureTest.test_team_has_no_ai_assistant -->
<!-- test: api.tests.TierFeatureTest.test_enterprise_has_everything -->
<!-- test: api.tests.MeEndpointTest.test_me_features_match_tier -->
<!-- test: accounts.tests.TierConstantsTest.test_tier_features_has_entry_for_each_tier -->
<!-- test: accounts.tests.TierConstantsTest.test_has_feature_free_vs_enterprise -->
<!-- test: accounts.tests.TierConstantsTest.test_has_feature_unknown_feature_returns_false -->
<!-- test: accounts.tests.TierConstantsTest.test_is_paid_tier -->
<!-- test: accounts.tests.UserModelTest.test_has_full_access_paid_tiers -->
<!-- test: accounts.tests.UserModelTest.test_has_ai_assistant_enterprise_only -->
<!-- test: accounts.tests.UserModelTest.test_can_collaborate_team_and_enterprise -->
<!-- test: accounts.tests.TierGatingScenarioTest.test_cross_tier_feature_matrix -->

| Feature | FREE | FOUNDER | PRO | TEAM | ENTERPRISE |
|---------|------|---------|-----|------|------------|
| basic_dsw | Y | Y | Y | Y | Y |
| basic_ml | - | Y | Y | Y | Y |
| full_tools | - | Y | Y | Y | Y |
| forge_api | - | Y | Y | Y | Y |
| collaboration | - | - | - | Y | Y |
| ai_assistant | - | - | - | - | Y |
| hoshin_kanri | - | - | - | - | Y |
| priority_support | - | Y | - | Y | Y |

### **4.3 Founder Pricing**

<!-- assert: Founder tier users retain locked pricing indefinitely | check=bill-founder-lock -->
<!-- impl: accounts/models.py -->
<!-- test: api.tests.BillingStatusTest.test_founder_availability_is_public -->

<!-- assert: is_founder_locked=True prevents tier reassignment during subscription sync | check=bill-founder-enforcement -->

Users with `is_founder_locked=True` keep $19/mo pricing regardless of plan changes. Maximum 50 founder slots. `get_founder_availability()` returns slots used/remaining.

---

## **5. PERMISSION DECORATORS**

### **5.1 Decorator Reference**

<!-- assert: All API endpoints use appropriate permission decorators | check=bill-decorators -->
<!-- impl: accounts/permissions.py -->
<!-- test: api.tests.PermissionDecoratorTest.test_rate_limited_blocks_unauthenticated -->
<!-- test: api.tests.PermissionDecoratorTest.test_rate_limited_allows_free_user -->
<!-- test: api.tests.PermissionDecoratorTest.test_gated_paid_blocks_free_user -->
<!-- test: api.tests.PermissionDecoratorTest.test_gated_paid_allows_pro_user -->
<!-- test: api.tests.PermissionDecoratorTest.test_require_team_blocks_pro -->
<!-- test: api.tests.PermissionDecoratorTest.test_require_team_allows_team -->
<!-- test: api.tests.QueryLimitTest.test_free_user_limited_to_5 -->
<!-- test: api.tests.QueryLimitTest.test_queries_increment_on_success -->
<!-- test: core.tests.TierGatingIntegrationTest.test_each_tier_create_access -->
<!-- test: core.tests.TierGatingIntegrationTest.test_can_create_org_flag_per_tier -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_unauthenticated_rate_limited_endpoint_returns_401 -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_free_user_rate_limited_succeeds_initially -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_free_user_exceeds_daily_limit_gets_429 -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_free_user_gated_paid_endpoint_returns_403 -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_pro_user_gated_paid_endpoint_succeeds -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_free_user_require_enterprise_returns_error -->
<!-- test: accounts.tests.PermissionDecoratorTest.test_enterprise_user_require_enterprise_passes_gate -->

| Decorator | Requires | Returns on Failure |
|-----------|----------|--------------------|
| `@require_auth` | Authenticated user | 401 |
| `@rate_limited` | Auth + daily limit check | 429 if limit reached |
| `@require_paid` | Any paid tier | 403 |
| `@require_team` | Team or Enterprise | 403 |
| `@require_enterprise` | Enterprise only | 403 |
| `@require_feature(f)` | Specific feature enabled | 403 |
| `@require_ml` | ML features (PRO+) | 403 |
| `@gated_paid` | Auth + full_tools + rate limit | 403 |
| `@allow_guest` | Guest token OR gated_paid | Guest or paid |
| `@require_org_admin` | Org admin role | 403 |
| `@gated` | Alias for `@rate_limited` | Legacy |

### **5.2 Rate Limiting Behavior**

`@rate_limited` checks `user.can_query()` before execution and calls `user.increment_queries()` atomically (F expression) on success (status < 400). This prevents counting failed requests against the daily limit.

---

## **6. STRIPE INTEGRATION**

### **6.1 Checkout Flow**

<!-- assert: Stripe Checkout used for all payment collection | check=bill-checkout -->
<!-- impl: accounts/billing.py -->
<!-- test: api.tests.BillingStatusTest.test_checkout_requires_auth -->

1. User hits `GET /billing/checkout/?plan=pro&region=in`
2. Server validates: no active subscription, founder slots available (if applicable)
3. Creates `stripe.checkout.Session` with price ID for region
4. Redirects user to Stripe-hosted checkout page
5. On success: Stripe fires `checkout.session.completed` webhook

Trial periods: Team and Enterprise get 14-day free trial. Others: no trial.

### **6.2 Customer Portal**

`GET /billing/portal/` creates a `stripe.billing_portal.Session` for self-service management. Users can view invoices, update payment method, or cancel subscription.

### **6.3 Subscription Sync**

<!-- assert: Subscription state synced from Stripe via webhooks | check=bill-sync -->
<!-- impl: accounts/billing.py -->

`sync_subscription_from_stripe(subscription_id)`:
1. Retrieves subscription from Stripe API
2. Creates or updates local `Subscription` record
3. Maps price ID to tier via `PRICE_TO_TIER`
4. Updates user tier based on subscription status
5. If active/trialing: set tier, `subscription_active=True`
6. If not active: downgrade to FREE

---

## **7. WEBHOOK SECURITY**

### **7.1 Signature Verification**

<!-- assert: All Stripe webhooks verified via HMAC-SHA256 signature | check=bill-webhook-sig -->
<!-- impl: accounts/billing.py -->

<!-- code: correct -->
```python
event = stripe.Webhook.construct_event(
    payload=request.body,
    sig_header=request.META.get("HTTP_STRIPE_SIGNATURE"),
    secret=settings.STRIPE_WEBHOOK_SECRET,
)
```

Invalid signatures return HTTP 400. Unsigned webhooks are never processed.

### **7.2 Handled Events**

<!-- assert: Webhook handler processes subscription lifecycle events | check=bill-webhook-events -->
<!-- impl: accounts/billing.py -->

| Event | Action |
|-------|--------|
| `checkout.session.completed` | Sync new subscription |
| `customer.subscription.updated` | Sync subscription changes |
| `customer.subscription.deleted` | Downgrade to FREE |
| `invoice.payment_failed` | Downgrade to FREE immediately |
| `invoice.paid` | Sync subscription (renewal) |

**Critical:** `invoice.payment_failed` immediately downgrades to FREE. No grace period for failed payments.

---

## **8. REGIONAL PRICING**

### **8.1 Supported Regions**

<!-- assert: Regional pricing covers 14+ currency regions | check=bill-regional -->
<!-- impl: accounts/billing.py -->
<!-- test: forge.tests.PricingCalculationTest.test_tabular_base_price -->
<!-- test: forge.tests.PricingCalculationTest.test_text_base_price -->
<!-- test: forge.tests.PricingCalculationTest.test_premium_multiplier -->
<!-- test: forge.tests.PricingCalculationTest.test_volume_discount_10k -->
<!-- test: forge.tests.PricingCalculationTest.test_volume_discount_50k -->
<!-- test: forge.tests.PricingCalculationTest.test_volume_discount_100k -->

| Region | Currency | Pro Price |
|--------|----------|-----------|
| US/EU/AU | USD | $49 |
| India | INR | 1,499 |
| Vietnam | VND | 349,000 |
| Ukraine | UAH | 349 |
| Philippines | PHP | 1,290 |
| Malaysia | MYR | 99 |
| Indonesia | IDR | 249,000 |
| Mexico | MXN | 449 |
| GCC/Middle East | AED | 149 |
| South Africa | ZAR | 349 |
| Kenya | KES | 1,990 |
| Nigeria | NGN | 4,990 |
| Brazil | BRL | 99 |
| Colombia | COP | 59,900 |
| Thailand | THB | 749 |

### **8.2 Region Detection**

Country detected from Cloudflare `CF-IPCountry` header (`HTTP_CF_IPCOUNTRY`). Falls back to USD if region not found or not supported.

Countries map to regions via `COUNTRY_TO_REGION` (e.g., GCC countries grouped under `ae` region). Each region has separate Stripe price IDs.

---

## **9. PAYMENT DATA SECURITY**

### **9.1 Stripe Customer ID Encryption**

<!-- assert: Stripe customer IDs encrypted at rest with indexed hash for lookups | check=bill-encryption -->
<!-- impl: accounts/models.py -->

| Field | Storage | Purpose |
|-------|---------|---------|
| `stripe_customer_id` | EncryptedCharField (Fernet) | Encrypted at rest |
| `stripe_customer_id_hash` | SHA-256 hash, indexed | Fast lookups without decryption |

**Pattern:** Store encrypted for decryption when needed for Stripe API calls. Store hash for indexed lookups (e.g., finding user by Stripe customer).

### **9.2 PCI DSS Compliance**

All card data handled by Stripe. Svend never sees, stores, or transmits card numbers. PCI DSS scope is limited to:
- Storing encrypted Stripe customer IDs
- Verifying webhook signatures
- Redirecting to Stripe-hosted checkout

### **9.3 Email Verification Tokens**

Generated with `secrets.token_urlsafe(32)`. Only SHA-256 hash stored in DB. Plaintext returned once in verification email. Comparison: hash input, compare to stored hash.

---

## **10. SUBSCRIPTION MODEL**

### **10.1 Data Model**

<!-- assert: Subscription model tracks Stripe state with proper lifecycle | check=bill-model -->
<!-- impl: accounts/models.py -->
<!-- test: api.tests.BillingStatusTest.test_billing_status_returns_tier_info -->
<!-- test: api.tests.LoginTest.test_login_returns_tier_and_limit -->
<!-- test: accounts.tests.SubscriptionModelTest.test_active_subscription_is_active -->
<!-- test: accounts.tests.SubscriptionModelTest.test_trialing_subscription_is_active -->
<!-- test: accounts.tests.SubscriptionModelTest.test_canceled_subscription_is_not_active -->
<!-- test: accounts.tests.SubscriptionModelTest.test_past_due_subscription_is_not_active -->
<!-- test: accounts.tests.SubscriptionModelTest.test_incomplete_subscription_is_not_active -->
<!-- test: accounts.tests.SubscriptionModelTest.test_paused_subscription_is_not_active -->
<!-- test: accounts.tests.UserModelTest.test_can_query_under_limit -->
<!-- test: accounts.tests.UserModelTest.test_can_query_at_limit -->
<!-- test: accounts.tests.UserModelTest.test_can_query_resets_when_past_reset_time -->
<!-- test: accounts.tests.UserModelTest.test_increment_queries_atomic -->
<!-- test: accounts.tests.TierGatingScenarioTest.test_full_tier_lifecycle -->
<!-- test: accounts.tests.TierGatingScenarioTest.test_rate_limit_lifecycle -->

`Subscription` (OneToOne to User):
- `stripe_subscription_id` (unique, indexed)
- `stripe_price_id` (plan identifier)
- `status` (ACTIVE, PAST_DUE, CANCELED, INCOMPLETE, TRIALING, UNPAID, PAUSED)
- `current_period_start`, `current_period_end`
- `cancel_at_period_end`
- `created_at`, `updated_at`

Property: `is_active` returns True if ACTIVE or TRIALING.

### **10.2 User Tier Fields**

User model tier fields:
- `tier` (FREE/FOUNDER/PRO/TEAM/ENTERPRISE)
- `queries_today`, `queries_reset_at` (daily limit tracking)
- `daily_limit` property (computed from tier)
- `can_query()` method (checks limit + resets if past reset time)
- `increment_queries()` (atomic F-expression increment)

---

## **11. BILLING ENDPOINTS**

### **11.1 URL Map**

<!-- assert: Billing endpoints registered with proper authentication | check=bill-endpoints -->
<!-- impl: accounts/urls.py -->
<!-- impl: accounts/billing.py -->
<!-- test: api.tests.BillingStatusTest.test_billing_status_requires_auth -->
<!-- test: api.tests.BillingStatusTest.test_billing_status_returns_tier_info -->
<!-- test: api.tests.BillingStatusTest.test_checkout_requires_auth -->
<!-- test: api.tests.BillingStatusTest.test_portal_requires_auth -->
<!-- test: api.tests.BillingStatusTest.test_founder_availability_is_public -->

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/billing/checkout/` | GET | login_required | Create Stripe checkout session |
| `/billing/portal/` | GET | login_required | Create Stripe portal session |
| `/billing/success/` | GET | login_required | Post-checkout success page |
| `/billing/cancel/` | GET | login_required | Post-checkout cancel page |
| `/billing/status/` | GET | login_required | Current subscription status |
| `/billing/founder-availability/` | GET | public | Founder slots remaining |
| `/webhooks/stripe/` | POST | csrf_exempt | Stripe webhook (4 URL aliases) |

### **11.2 Webhook URL Aliases**

Four aliases handle Stripe's POST format variations: `/webhooks/stripe/`, `/webhooks/stripe`, `/billing/webhook/`, `/billing/webhook`. All route to the same handler.

---

## **12. INVITE CODE SYSTEM**

### **12.1 Alpha Access Gating**

<!-- assert: Invite codes gate registration when SVEND_REQUIRE_INVITE is set | check=bill-invites -->
<!-- impl: accounts/models.py -->
<!-- test: accounts.tests.InviteCodeTest.test_generate_and_use -->
<!-- test: accounts.tests.InviteCodeTest.test_use_exhausted_code_fails -->
<!-- test: accounts.tests.InviteCodeTest.test_generate_format -->
<!-- test: accounts.tests.InviteCodeTest.test_deactivated_code_is_not_valid -->

`InviteCode` model:
- `code` (format: "XXXX-XXXX", crypto-secure hex)
- `max_uses`, `times_used`, `is_active`
- `used_by` (ManyToMany to User)
- `is_valid` property (active AND uses remaining)

Generation: `InviteCode.generate(count, max_uses, note)`. Management command: `python manage.py generate_invites`.

---

## **13. ANTI-PATTERNS**

### **13.1 Storing Card Data**

**PROHIBITED:** Storing, logging, or transmitting card numbers. All payment data stays with Stripe.

### **13.2 Bypassing Feature Gates**

<!-- code: prohibited -->
```python
# Checking tier directly instead of using decorator
if request.user.tier in ["PRO", "TEAM", "ENTERPRISE"]:
    do_premium_thing()
```

<!-- code: correct -->
```python
@gated_paid
def my_view(request):
    do_premium_thing()
```

Use decorators. They handle auth, rate limiting, and query counting atomically.

### **13.3 Processing Unsigned Webhooks**

**PROHIBITED:** Skipping `stripe.Webhook.construct_event()` signature verification. Every webhook MUST be verified.

### **13.4 Grace Period on Failed Payment**

<!-- assert: Failed payment (invoice.payment_failed) immediately downgrades user to FREE tier — no grace period | check=bill-no-grace-period -->

**PROHIBITED:** Allowing continued access after `invoice.payment_failed`. The current implementation correctly downgrades to FREE immediately.

### **13.5 Hardcoding Price IDs**

**PROHIBITED:** Scattering Stripe price IDs across views. All price mappings MUST live in `billing.py:PRICE_TO_TIER` and `REGIONAL_PRICES`.

---

## **14. ACCEPTANCE CRITERIA**

<!-- table: acceptance-criteria -->
| Criterion | Validation Method |
|-----------|-------------------|
| Five tiers defined with correct limits | `constants.py` audit |
| All premium endpoints use permission decorators | grep for `@gated_paid`, `@require_paid`, etc. |
| Webhook signature verified on every event | `billing.py` code audit |
| Stripe customer ID encrypted at rest | Model field inspection |
| Payment failure immediately downgrades | `invoice.payment_failed` handler audit |
| Regional pricing covers 14+ regions | `REGIONAL_PRICES` dict length |
| Founder lock prevents tier reassignment | `is_founder_locked` logic audit |
| Daily limit enforced via `can_query()` | Permission decorator trace |
| Query increment uses F-expression (atomic) | `increment_queries()` code audit |
| No card data stored in database | Full model audit |
| Checkout uses Stripe-hosted page | `create_checkout_session()` audit |
| All price IDs centralized in billing.py | grep for `price_` across codebase |

---

## **15. COMPLIANCE MAPPING**

<!-- table: compliance-mapping -->
<!-- control: SOC 2 CC6.1 -->
<!-- control: SOC 2 CC6.7 -->
<!-- control: PCI DSS 3.4 -->

| Requirement | External Standard | BILL-001 Section |
|-------------|-------------------|------------------|
| Logical access controls | SOC 2 CC6.1 | §5 (Permission Decorators) |
| Restriction of access | SOC 2 CC6.7 | §4 (Tier Matrix, Feature Gating) |
| PAN unreadable | PCI DSS 3.4 | §9.2 (Delegated to Stripe) |
| Data encryption at rest | SOC 2 CC6.1 | §9.1 (Encrypted Stripe IDs) |
| Webhook integrity | SOC 2 CC6.2 | §7 (Signature Verification) |
| Right to deletion | GDPR Art. 17 | §10 (Subscription Model) |
| Pricing transparency | — | §8 (Regional Pricing) |

---

## **16. BILLING API COVERAGE**

### **16.1 Tier Constants & Feature Gating**

<!-- assert: Tier constants correctly gate features by subscription level | check=bill-tier-constants -->
<!-- impl: accounts/constants.py:Tier -->
<!-- impl: accounts/constants.py:get_daily_limit -->
<!-- impl: accounts/constants.py:has_feature -->
<!-- impl: accounts/constants.py:is_paid_tier -->
<!-- impl: accounts/constants.py:can_use_anthropic -->
<!-- impl: accounts/constants.py:can_use_ml -->
<!-- impl: accounts/constants.py:can_use_tools -->
<!-- impl: accounts/constants.py:get_founder_availability -->
<!-- impl: accounts/constants.py:Industry -->
<!-- impl: accounts/constants.py:Role -->
<!-- impl: accounts/constants.py:ExperienceLevel -->
<!-- impl: accounts/constants.py:OrganizationSize -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_get_daily_limit_returns_correct_values_per_tier -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_has_feature_enterprise_has_ai_assistant -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_has_feature_collaboration_for_team_and_enterprise -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_is_paid_tier_classifies_correctly -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_can_use_anthropic_matches_paid_tiers -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_can_use_ml_requires_basic_ml_feature -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_can_use_tools_requires_full_tools_feature -->
<!-- test: accounts.tests_coverage.TierConstantsTest.test_get_founder_availability_counts_founder_users -->

### **16.2 Billing Views**

<!-- assert: Billing endpoints handle checkout flow, subscription status, and founder availability | check=bill-views -->
<!-- impl: accounts/billing.py:subscription_status -->
<!-- impl: accounts/billing.py:founder_availability -->
<!-- impl: accounts/billing.py:checkout_success -->
<!-- impl: accounts/billing.py:checkout_cancel -->
<!-- impl: accounts/billing.py:create_checkout_session -->
<!-- impl: accounts/billing.py:create_portal_session -->
<!-- impl: accounts/billing.py:get_price_for_region -->
<!-- impl: accounts/billing.py:get_or_create_stripe_customer -->
<!-- impl: accounts/billing.py:add_org_seat -->
<!-- impl: accounts/billing.py:remove_org_seat -->
<!-- impl: accounts/billing.py:sync_subscription_from_stripe -->
<!-- impl: accounts/billing.py:stripe_webhook -->
<!-- test: accounts.tests_coverage.BillingViewsTest.test_subscription_status_returns_tier_and_limits -->
<!-- test: accounts.tests_coverage.BillingViewsTest.test_subscription_status_includes_subscription_detail -->
<!-- test: accounts.tests_coverage.BillingViewsTest.test_founder_availability_public_endpoint -->
<!-- test: accounts.tests_coverage.BillingViewsTest.test_checkout_success_syncs_subscription -->
<!-- test: accounts.tests_coverage.BillingViewsTest.test_checkout_success_rejects_customer_mismatch -->
<!-- test: accounts.tests_coverage.BillingViewsTest.test_checkout_cancel_redirects_to_settings -->

### **16.3 Subscription & Invite Code Models**

<!-- assert: Subscription model tracks active/inactive state; InviteCode gates registration | check=bill-models -->
<!-- impl: accounts/models.py:Subscription -->
<!-- impl: accounts/models.py:InviteCode -->
<!-- impl: accounts/models.py:LoginAttempt -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_subscription_is_active_for_active_status -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_subscription_is_not_active_for_canceled -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_invite_code_is_valid_when_unused -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_invite_code_use_increments_count_and_links_user -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_invite_code_exhausted_returns_false -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_invite_code_generate_creates_codes -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_login_attempt_lockout_after_max_failures -->
<!-- test: accounts.tests_coverage.AccountModelsTest.test_login_attempt_clear_on_success -->

### **16.4 Middleware**

<!-- assert: Subscription middleware enforces tier on every request; QueryLimit middleware rate-limits chat | check=bill-middleware -->
<!-- impl: accounts/middleware.py:SubscriptionMiddleware -->
<!-- impl: accounts/middleware.py:QueryLimitMiddleware -->
<!-- impl: accounts/middleware.py:NoCacheDynamicMiddleware -->
<!-- impl: accounts/middleware.py:SiteVisitMiddleware -->
<!-- test: accounts.tests_coverage.MiddlewareTest.test_subscription_middleware_sets_flags_for_paid_user -->
<!-- test: accounts.tests_coverage.MiddlewareTest.test_subscription_middleware_updates_last_active -->
<!-- test: accounts.tests_coverage.MiddlewareTest.test_query_limit_middleware_blocks_exhausted_user -->
<!-- test: accounts.tests_coverage.MiddlewareTest.test_query_limit_middleware_allows_under_limit -->
<!-- test: accounts.tests_coverage.MiddlewareTest.test_site_visit_middleware_skips_staff -->
<!-- test: accounts.tests_coverage.MiddlewareTest.test_site_visit_middleware_skips_api_paths -->

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-03 | Eric + Claude | Initial release — documents billing, tiers, Stripe, and feature gating |
| 1.1 | 2026-03-07 | Claude | Add §16 symbol-level impl/test hooks for billing API coverage |

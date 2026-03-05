# Gmail SMTP — Transactional Email

**Purpose:** Account verification, password resets, onboarding drips, email campaigns.
**Provider:** Gmail (Google Workspace)

---

## How It Works

```
Django send_mail() → SMTP (smtp.gmail.com:587, TLS) → Gmail delivers to recipient
```

## Credentials

| Secret | Location | Purpose |
|--------|----------|---------|
| `EMAIL_HOST_USER` | `svend_config/config.py` | SMTP username (`hello@svend.ai`) |
| `EMAIL_HOST_PASSWORD` | `svend_config/config.py` | Gmail app-specific password |

**This is a Gmail app password, not the account password.** Generated in Google Account → Security → App Passwords.

## Configuration

| Setting | Value |
|---------|-------|
| Host | `smtp.gmail.com` |
| Port | `587` |
| TLS | Enabled |
| From | `Svend <hello@svend.ai>` |

## Integration Points

| File | Purpose |
|------|---------|
| `svend/settings.py` | Django email backend configuration |
| `svend_config/config.py` | SMTP credentials |
| `api/views.py` | Email campaign system, onboarding drips |
| `accounts/views.py` | Verification emails, password reset |
| `templates/registration/` | Password reset email templates |

## Emails We Send

| Type | Trigger | Template |
|------|---------|----------|
| Email verification | Registration | Inline in view |
| Password reset | User request | `registration/password_reset_email.html` |
| Onboarding: Welcome | Registration + 0h | Campaign system |
| Onboarding: Getting Started | Registration + 24h | Campaign system |
| Onboarding: Tips | Registration + 72h | Campaign system |
| Onboarding: Learning Path | Registration + 168h | Campaign system |

## Common Tasks

### Test email sending
```bash
python3 manage.py shell
>>> from django.core.mail import send_mail
>>> send_mail("Test", "Body", "hello@svend.ai", ["your@email.com"])
```

### Check if Gmail is blocking sends
Google sometimes blocks SMTP if it detects unusual activity. Check:
- https://myaccount.google.com/security → Recent security activity
- Ensure "Less secure app access" or app passwords are configured

## Gmail Limits

- **Daily send limit:** 500 emails/day (free Gmail) or 2,000/day (Workspace)
- **Per-minute:** ~20 messages
- If limits are hit, emails silently fail. Consider migrating to a transactional email service (Resend, Postmark) if volume grows.

## Fallback

`svend_config/config.py` has Resend (`smtp.resend.com`) as a default fallback option. Not currently active but available if Gmail becomes unreliable.

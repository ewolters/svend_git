# Svend Alpha Access System

## Overview

Alpha access is controlled via invite codes. Users must have a valid invite code to register.

**To disable invite requirement (open registration):** Set `SVEND_REQUIRE_INVITE=false` in `.env`

---

## User Flow

1. User visits svend.ai
2. Sees **"SVEND ALPHA"** registration form
3. Enters their invite code (formats automatically as `XXXX-XXXX`)
4. Fills in username, optional email, password (8+ chars)
5. Clicks **"Create Account"**
6. On success, automatically logged in and taken to chat
7. Returning users click **"Sign in"** to switch to login form

---

## Invite Code Management

### Generate Codes

```bash
# Generate single-use codes
python manage.py generate_invites 5 --note "Friends & Family Alpha"

# Generate multi-use code (e.g., for a group)
python manage.py generate_invites 1 --uses 10 --note "Office group"

# Just one code, no note
python manage.py generate_invites
```

### List All Codes

```bash
python manage.py generate_invites --list
```

Output:
```
Code            Uses       Status     Note
------------------------------------------------------------
7374-0CB8       1/1        exhausted  Friends & Family Alpha
DEA6-DC9B       0/1        valid      Friends & Family Alpha
...
```

### Code Format

Codes are 8 characters in `XXXX-XXXX` format (e.g., `7374-0CB8`).
Case-insensitive when entered by users.

---

## Registration API

**Endpoint:** `POST /api/register/`

**Request:**
```json
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "securepassword123",
  "invite_code": "DEA6-DC9B"
}
```

**Response (success):**
```json
{
  "status": "registered",
  "username": "newuser",
  "tier": "beta",
  "message": "Welcome to Svend alpha!"
}
```

**Response (errors):**
- `400` - Invalid/missing invite code, username taken, password too short
- No invite code provided when required

---

## Registered Alpha Users

| Username | Email | Invite Code | Registered |
|----------|-------|-------------|------------|
| princess_peach | britt.soper0504@gmail.com | 7374-0CB8 | 2026-01-18 |
| testuser | (test account) | N/A | pre-created |

---

## Available Invite Codes

| Code | Max Uses | Status | Note |
|------|----------|--------|------|
| 7374-0CB8 | 1 | USED | Friends & Family Alpha (princess_peach) |
| DEA6-DC9B | 1 | available | Friends & Family Alpha |
| 378B-5D25 | 1 | available | Friends & Family Alpha |
| 8ACA-84FD | 1 | available | Friends & Family Alpha |
| E0BD-61ED | 1 | available | Friends & Family Alpha |

---

## Database Model

**Table:** `invite_codes`

| Field | Type | Description |
|-------|------|-------------|
| code | CharField(20) | The invite code (unique, indexed) |
| max_uses | IntegerField | Maximum times code can be used |
| times_used | IntegerField | Current usage count |
| is_active | BooleanField | Can be deactivated manually |
| note | CharField(255) | Optional description |
| created_at | DateTimeField | When code was generated |
| used_by | M2M to User | Users who used this code |

---

## Deactivating a Code

Via Django admin or shell:

```python
from accounts.models import InviteCode
code = InviteCode.objects.get(code='XXXX-XXXX')
code.is_active = False
code.save()
```

---

## Switching to Open Registration

When ready for public beta:

1. Edit `/home/eric/Desktop/svend_transfer/svend/.env`
2. Add or change: `SVEND_REQUIRE_INVITE=false`
3. Restart gunicorn

Registration will then work without invite codes.

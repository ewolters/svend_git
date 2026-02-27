"""Encrypt existing Stripe customer IDs and hash verification tokens.

For stripe_customer_id: encrypts the value and populates the hash column.
For email_verification_token: hashes existing plaintext tokens.
"""

import hashlib

from django.db import migrations


def hash_value(value):
    """SHA-256 hash for lookups."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def encrypt_users(apps, schema_editor):
    User = apps.get_model("accounts", "User")
    for user in User.objects.all():
        updated_fields = []

        # Populate stripe_customer_id_hash from plaintext
        if user.stripe_customer_id:
            # The ORM field will encrypt on save, but we need the hash
            # from the PLAINTEXT value. Since from_db_value has fallback,
            # reading gives us the plaintext.
            user.stripe_customer_id_hash = hash_value(user.stripe_customer_id)
            updated_fields.append("stripe_customer_id_hash")

        # Hash existing plaintext verification tokens
        if user.email_verification_token and len(user.email_verification_token) < 64:
            # Still plaintext (SHA-256 hex is exactly 64 chars)
            user.email_verification_token = hash_value(user.email_verification_token)
            updated_fields.append("email_verification_token")

        if updated_fields:
            # Full save to also encrypt stripe_customer_id via ORM
            user.save()


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0009_user_stripe_customer_id_hash_and_more"),
    ]

    operations = [
        migrations.RunPython(encrypt_users, migrations.RunPython.noop),
    ]

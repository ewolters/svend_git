"""Override Django's flush command to prevent accidental production database wipe.

This command blocks flush on the default (production) database. Only the test
database (managed by Django's test runner) is allowed. To force a flush in
non-production environments, pass --i-know-what-i-am-doing.

Incident context: On 2026-03-07, `call_command('flush', '--no-input')` was run
with DJANGO_SETTINGS_MODULE=svend.settings, which wiped the production database.
This safeguard ensures it never happens again.
"""

import sys

from django.core.management.commands.flush import Command as BaseFlush


class Command(BaseFlush):
    help = "BLOCKED: use manage.py test for test isolation. See core/management/commands/flush.py."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--i-know-what-i-am-doing",
            action="store_true",
            dest="force_allow",
            help="Bypass production safety check (DANGEROUS).",
        )

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        # Allow if running inside Django's test runner (db name starts with 'test_')
        from django.conf import settings

        db_name = settings.DATABASES.get(database, {}).get("NAME", "")
        is_test_db = str(db_name).startswith("test_") or "test" in str(db_name).lower()

        if not is_test_db and not force:
            self.stderr.write(
                self.style.ERROR(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  BLOCKED: flush on production database is prohibited.       ║\n"
                    "║                                                             ║\n"
                    "║  • Use `manage.py test` for test isolation (auto-creates    ║\n"
                    "║    and destroys a test database).                           ║\n"
                    "║  • If you REALLY need this, pass --i-know-what-i-am-doing   ║\n"
                    "║                                                             ║\n"
                    "║  Incident: 2026-03-07 production DB wipe via flush.         ║\n"
                    "╚══════════════════════════════════════════════════════════════╝\n"
                )
            )
            sys.exit(1)

        super().handle(*args, **options)

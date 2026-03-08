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

from ._safety import _is_production_db, add_force_flag


class Command(BaseFlush):
    help = "BLOCKED: use manage.py test for test isolation. See core/management/commands/flush.py."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            self.stderr.write(
                self.style.ERROR(
                    "\n"
                    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
                    "\u2551  BLOCKED: flush on production database is prohibited.       \u2551\n"
                    "\u2551                                                             \u2551\n"
                    "\u2551  \u2022 Use `manage.py test` for test isolation (auto-creates    \u2551\n"
                    "\u2551    and destroys a test database).                           \u2551\n"
                    "\u2551  \u2022 If you REALLY need this, pass --i-know-what-i-am-doing   \u2551\n"
                    "\u2551                                                             \u2551\n"
                    "\u2551  Incident: 2026-03-07 production DB wipe via flush.         \u2551\n"
                    "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n"
                )
            )
            sys.exit(1)

        super().handle(*args, **options)

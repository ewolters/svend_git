"""Override Django's testserver to block on production database.

testserver calls loaddata internally and runs a development server —
it should never be run against a production database.
"""

import sys

from django.core.management.commands.testserver import Command as BaseTestServer

from ._safety import _db_name, _is_production_db, add_force_flag, blocked_banner


class Command(BaseTestServer):
    help = "BLOCKED on production. Runs dev server with fixture data — use only on test/dev."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            self.stderr.write(
                self.style.ERROR(blocked_banner("testserver", _db_name(database)))
            )
            sys.exit(1)

        super().handle(*args, **options)

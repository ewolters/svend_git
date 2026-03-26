"""Override Django's loaddata to block on production database.

loaddata overwrites rows by primary key — running it against production
can silently corrupt or replace live data.
"""

import sys

from django.core.management.commands.loaddata import Command as BaseLoadData

from ._safety import _db_name, _is_production_db, add_force_flag, blocked_banner


class Command(BaseLoadData):
    help = "BLOCKED on production. Loads fixture data — use only on test/dev databases."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            self.stderr.write(
                self.style.ERROR(blocked_banner("loaddata", _db_name(database)))
            )
            sys.exit(1)

        super().handle(*args, **options)

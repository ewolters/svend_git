"""Override Django's squashmigrations to block on production.

squashmigrations rewrites migration history. This should be done on a dev
machine and the result deployed — never run directly on production.
"""

import sys

from django.core.management.commands.squashmigrations import (
    Command as BaseSquashMigrations,
)

from ._safety import _db_name, _is_production_db, add_force_flag, blocked_banner


class Command(BaseSquashMigrations):
    help = "BLOCKED on production. Squashes migrations — run on dev, deploy the result."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        # squashmigrations doesn't have a --database flag, always uses default
        database = "default"
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            self.stderr.write(self.style.ERROR(blocked_banner("squashmigrations", _db_name(database))))
            sys.exit(1)

        super().handle(*args, **options)

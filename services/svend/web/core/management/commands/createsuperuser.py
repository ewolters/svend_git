"""Override Django's createsuperuser to block on production database.

createsuperuser creates an admin account bypassing the normal invite flow
and permission controls. On production, use the admin panel or invite system.
"""

import sys

from django.contrib.auth.management.commands.createsuperuser import (
    Command as BaseCreateSuperUser,
)

from ._safety import _db_name, _is_production_db, add_force_flag, blocked_banner


class Command(BaseCreateSuperUser):
    help = "BLOCKED on production. Creates superuser — use admin panel or invite flow instead."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            self.stderr.write(self.style.ERROR(blocked_banner("createsuperuser", _db_name(database))))
            sys.exit(1)

        super().handle(*args, **options)

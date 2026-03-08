"""Override Django's sqlflush to block on production database.

sqlflush generates DELETE/TRUNCATE SQL. While it doesn't execute the SQL directly,
its output is commonly piped to dbshell or psql, making it a vector for accidental
production data loss.
"""

import sys

from django.core.management.commands.sqlflush import Command as BaseSqlFlush

from ._safety import _db_name, _is_production_db, add_force_flag, blocked_banner


class Command(BaseSqlFlush):
    help = "BLOCKED on production. Generates SQL to flush tables \u2014 use only on test databases."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            self.stderr.write(self.style.ERROR(blocked_banner("sqlflush", _db_name(database))))
            sys.exit(1)

        super().handle(*args, **options)

"""Override Django's sqlflush to block on production database.

sqlflush generates DELETE/TRUNCATE SQL. While it doesn't execute the SQL directly,
its output is commonly piped to dbshell or psql, making it a vector for accidental
production data loss.
"""

import sys

from django.core.management.commands.sqlflush import Command as BaseSqlFlush


class Command(BaseSqlFlush):
    help = "BLOCKED on production. Generates SQL to flush tables — use only on test databases."

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

        from django.conf import settings

        db_name = settings.DATABASES.get(database, {}).get("NAME", "")
        is_test_db = str(db_name).startswith("test_") or "test" in str(db_name).lower()

        if not is_test_db and not force:
            self.stderr.write(
                self.style.ERROR(
                    "\nBLOCKED: sqlflush on production database is prohibited.\n"
                    "Pass --i-know-what-i-am-doing to override.\n"
                )
            )
            sys.exit(1)

        super().handle(*args, **options)

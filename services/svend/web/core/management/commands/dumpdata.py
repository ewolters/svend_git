"""Override Django's dumpdata to warn about PII/secrets on production.

dumpdata exports database contents as JSON/YAML. On production, the output
may contain PII, secrets, or other sensitive data.
"""

from django.core.management.commands.dumpdata import Command as BaseDumpData

from ._safety import _db_name, _is_production_db


class Command(BaseDumpData):
    help = "Dumps data with a PII/secrets warning on production."

    def handle(self, *args, **options):
        database = options.get("database", "default")

        if _is_production_db(database):
            self.stderr.write(
                self.style.WARNING(
                    f"\n\u26a0 Dumping from production database: {_db_name(database)}\n"
                    "  Output may contain PII, secrets, or sensitive data.\n"
                    "  Do NOT commit, share, or upload the output.\n"
                )
            )

        super().handle(*args, **options)

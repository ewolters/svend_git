"""Override Django's shell to warn when connected to production.

The Django shell has full ORM access and can modify or delete any data.
This override shows which database is connected.
"""

from django.core.management.commands.shell import Command as BaseShell

from ._safety import _db_name, _is_production_db


class Command(BaseShell):
    help = "Opens Django shell with production database warning."

    def handle(self, *args, **options):
        if _is_production_db():
            self.stderr.write(
                self.style.WARNING(
                    f"\n\u26a0 Connected to production database: {_db_name()}\n"
                    "  ORM operations affect live data. Proceed with caution.\n"
                )
            )

        super().handle(*args, **options)

"""Override Django's makemigrations to show a CHG-001 reminder on production.

makemigrations generates migration files. On production, this reminds the
developer to follow the change management process before committing.
"""

from django.core.management.commands.makemigrations import (
    Command as BaseMakeMigrations,
)

from ._safety import _db_name, _is_production_db


class Command(BaseMakeMigrations):
    help = "Generates migrations with a CHG-001 process reminder on production."

    def handle(self, *args, **options):
        if _is_production_db():
            self.stderr.write(
                self.style.WARNING(
                    f"\n\u26a0 Running on production server (DB: {_db_name()}).\n"
                    "  Migration files require a ChangeRequest per CHG-001 \xa77.1.2.\n"
                    "  Do not commit migration files without an approved CR.\n"
                )
            )

        super().handle(*args, **options)

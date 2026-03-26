"""Override Django's dbshell to require explicit confirmation on production.

dbshell opens an interactive database shell where DROP, DELETE, TRUNCATE
commands can be executed without any guardrails.
"""

import sys

from django.core.management.commands.dbshell import Command as BaseDbShell

from ._safety import _db_name, _is_production_db, add_force_flag


class Command(BaseDbShell):
    help = "Opens database shell with production safety prompt."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        if _is_production_db(database) and not force:
            db = _db_name(database)
            self.stderr.write(
                self.style.WARNING(
                    f"\n\u26a0 WARNING: You are about to open a shell on the PRODUCTION database '{db}'.\n"
                    "Any DROP, DELETE, or TRUNCATE commands will be irreversible.\n"
                )
            )
            try:
                answer = input("Type 'production' to confirm: ")
            except (EOFError, KeyboardInterrupt):
                self.stderr.write("\nAborted.")
                sys.exit(1)

            if answer.strip().lower() != "production":
                self.stderr.write(
                    self.style.ERROR("Aborted. Did not type 'production'.")
                )
                sys.exit(1)

        super().handle(*args, **options)

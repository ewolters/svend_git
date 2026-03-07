"""Override Django's dbshell to require explicit confirmation on production.

dbshell opens an interactive database shell where DROP, DELETE, TRUNCATE
commands can be executed without any guardrails.
"""

import sys

from django.core.management.commands.dbshell import Command as BaseDbShell


class Command(BaseDbShell):
    help = "Opens database shell with production safety prompt."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--i-know-what-i-am-doing",
            action="store_true",
            dest="force_allow",
            help="Skip production safety confirmation.",
        )

    def handle(self, *args, **options):
        database = options.get("database", "default")
        force = options.get("force_allow", False)

        from django.conf import settings

        db_name = settings.DATABASES.get(database, {}).get("NAME", "")
        is_test_db = str(db_name).startswith("test_") or "test" in str(db_name).lower()

        if not is_test_db and not force:
            self.stderr.write(
                self.style.WARNING(
                    f"\n⚠ WARNING: You are about to open a shell on the PRODUCTION database '{db_name}'.\n"
                    "Any DROP, DELETE, or TRUNCATE commands will be irreversible.\n"
                )
            )
            try:
                answer = input("Type 'production' to confirm: ")
            except (EOFError, KeyboardInterrupt):
                self.stderr.write("\nAborted.")
                sys.exit(1)

            if answer.strip().lower() != "production":
                self.stderr.write(self.style.ERROR("Aborted. Did not type 'production'."))
                sys.exit(1)

        super().handle(*args, **options)

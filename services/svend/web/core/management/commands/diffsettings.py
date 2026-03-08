"""Override Django's diffsettings to warn about secret exposure.

diffsettings prints all settings that differ from defaults, which includes
SECRET_KEY, database passwords, API keys, and other sensitive values.
"""

from django.core.management.commands.diffsettings import Command as BaseDiffSettings

from ._safety import _is_production_db


class Command(BaseDiffSettings):
    help = "Shows settings diff with a secrets warning on production."

    def handle(self, *args, **options):
        if _is_production_db():
            self.stderr.write(
                self.style.WARNING(
                    "\n\u26a0 Output contains SECRET_KEY, database passwords, and API keys.\n"
                    "  Do NOT share, paste, or commit this output.\n"
                )
            )

        super().handle(*args, **options)

"""Override Django's migrate to require confirmation and audit on production.

migrate alters database schema. On production, this shows the database name,
pending migration count, and requires explicit confirmation. Successful runs
are logged to SysLogEntry for audit trail.
"""

import sys

from django.core.management.commands.migrate import Command as BaseMigrate
from django.db import connections

from ._safety import _db_name, _is_production_db, add_force_flag


class Command(BaseMigrate):
    help = "Applies migrations with production safety confirmation and audit logging."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        add_force_flag(parser)

    def handle(self, *args, **options):
        database = options.get("database", "default")

        if not _is_production_db(database):
            return super().handle(*args, **options)

        db = _db_name(database)
        force = options.get("force_allow", False)

        # Count pending migrations
        from django.db.migrations.executor import MigrationExecutor

        connection = connections[database]
        executor = MigrationExecutor(connection)
        plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
        count = len(plan)

        self.stderr.write(self.style.WARNING(f"\n\u26a0 Production database: {db}\n  Pending migrations: {count}\n"))

        if options.get("fake"):
            self.stderr.write(
                self.style.ERROR(
                    "  WARNING: --fake corrupts migration state.\n  Use only to recover from a known-good state.\n"
                )
            )

        if not force:
            try:
                answer = input("Type 'production' to confirm: ")
            except (EOFError, KeyboardInterrupt):
                self.stderr.write("\nAborted.")
                sys.exit(1)

            if answer.strip().lower() != "production":
                self.stderr.write(self.style.ERROR("Aborted. Did not type 'production'."))
                sys.exit(1)

        super().handle(*args, **options)

        # Audit log
        try:
            from syn.audit.utils import generate_entry

            generate_entry(
                tenant_id="system",
                actor="manage.py",
                event_name="management_command.migrate",
                payload={
                    "database": database,
                    "db_name": db,
                    "pending_count": count,
                    "fake": bool(options.get("fake")),
                    "forced": force,
                },
            )
        except Exception:
            # Don't fail the migration because audit logging failed
            self.stderr.write(self.style.WARNING("Warning: Could not create audit log entry."))

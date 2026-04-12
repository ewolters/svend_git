"""
Management command to verify audit log integrity.

Usage:
    python manage.py verify_syslog_integrity
    python manage.py verify_syslog_integrity --tenant=tenant-123
    python manage.py verify_syslog_integrity --all-tenants

Compliance: SOC 2 CC7.2 / ISO 27001 A.12.7
"""

from django.core.management.base import BaseCommand, CommandError

from syn.audit.models import SysLogEntry
from syn.audit.utils import record_integrity_violation, verify_chain_integrity


class Command(BaseCommand):
    help = "Verify audit log chain integrity for tenant(s)"

    def add_arguments(self, parser):
        parser.add_argument("--tenant", type=str, help="Tenant ID to verify (default: verify all)")

        parser.add_argument("--all-tenants", action="store_true", help="Explicitly verify all tenants")

        parser.add_argument(
            "--record-violations",
            action="store_true",
            help="Record detected violations to database",
        )

        parser.add_argument(
            "--fail-on-violation",
            action="store_true",
            help="Exit with error code if violations detected",
        )

    def handle(self, *args, **options):
        tenant_id = options.get("tenant")
        all_tenants = options.get("all_tenants")
        record_violations = options.get("record_violations")
        fail_on_violation = options.get("fail_on_violation")

        # Determine which tenants to verify
        if tenant_id:
            tenant_ids = [tenant_id]
            self.stdout.write(f"Verifying audit log for tenant: {tenant_id}")
        elif all_tenants:
            # Get all unique tenant IDs from audit log
            tenant_ids = SysLogEntry.objects.values_list("tenant_id", flat=True).distinct()
            self.stdout.write(f"Verifying audit logs for {len(tenant_ids)} tenants")
        else:
            # Default: verify all tenants
            tenant_ids = SysLogEntry.objects.values_list("tenant_id", flat=True).distinct()
            self.stdout.write(f"Verifying audit logs for {len(tenant_ids)} tenants")

        # Verify each tenant's chain
        total_violations = 0
        results = []

        for tid in tenant_ids:
            self.stdout.write(f"\nVerifying tenant: {tid}")

            result = verify_chain_integrity(tid)
            results.append((tid, result))

            # Display results
            if result["is_valid"]:
                self.stdout.write(self.style.SUCCESS(f"  ✓ Chain intact: {result['total_entries']} entries verified"))
            else:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Chain compromised: {len(result['violations'])} violations detected")
                )

                # Show violation details
                for violation in result["violations"]:
                    self.stdout.write(self.style.WARNING(f"    - {violation['type']}: {violation['message']}"))

                    # Record violation if requested
                    if record_violations:
                        try:
                            record_integrity_violation(
                                tenant_id=tid,
                                violation_type=violation["type"],
                                entry_id=violation.get("entry_id"),
                                details=violation,
                            )
                            self.stdout.write(self.style.WARNING("      Violation recorded and alert emitted"))
                        except Exception as e:
                            self.stdout.write(self.style.ERROR(f"      Failed to record violation: {e}"))

                total_violations += len(result["violations"])

        # Summary
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("VERIFICATION SUMMARY")
        self.stdout.write("=" * 70)

        valid_count = sum(1 for _, r in results if r["is_valid"])
        invalid_count = len(results) - valid_count
        total_entries = sum(r["total_entries"] for _, r in results)

        self.stdout.write(f"Tenants verified: {len(results)}")
        self.stdout.write(f"Total entries: {total_entries}")
        self.stdout.write(self.style.SUCCESS(f"Valid chains: {valid_count}"))

        if invalid_count > 0:
            self.stdout.write(self.style.ERROR(f"Invalid chains: {invalid_count}"))
            self.stdout.write(self.style.ERROR(f"Total violations: {total_violations}"))

        # Exit with error if violations detected and flag set
        if fail_on_violation and total_violations > 0:
            raise CommandError(f"Integrity verification failed: {total_violations} violations detected")

        if total_violations == 0:
            self.stdout.write(self.style.SUCCESS("\n✓ All audit log chains are intact"))
        else:
            self.stdout.write(self.style.ERROR("\n✗ Integrity violations detected. Investigate immediately!"))

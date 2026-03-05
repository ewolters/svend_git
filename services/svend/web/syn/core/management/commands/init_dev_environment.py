"""
Development Environment Initialization Command - BOOT-001 / PROD-BOOT-005 Compliant

Automates the full boot sequence for a fresh database in development environments.
This resolves PROD-BOOT-005: "No automated dev environment initialization"

Boot Sequence (in order):
1. Run migrations (if needed)
2. Load app definitions from YAML
3. Create default tenant (if none exists)
4. Create superuser with tenant assignment
5. Initialize apps for tenant
6. Seed primitive registry (PRM-001 §8)
7. Seed reflexes (REF-001)

Usage:
    python manage.py init_dev_environment                    # Full initialization
    python manage.py init_dev_environment --skip-migrate     # Skip migrations
    python manage.py init_dev_environment --tenant-only      # Only create tenant
    python manage.py init_dev_environment --no-input         # Non-interactive mode

Security Note:
    This command is for DEVELOPMENT ENVIRONMENTS ONLY.
    It creates users with default credentials which should be changed.
    Production environments should use `tenant provision` instead.
"""

import secrets
import string

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

User = get_user_model()


class Command(BaseCommand):
    help = "Initialize development environment with all required data (BOOT-001 / PROD-BOOT-005 compliant)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-migrate",
            action="store_true",
            help="Skip running migrations"
        )
        parser.add_argument(
            "--skip-apps",
            action="store_true",
            help="Skip loading app definitions"
        )
        parser.add_argument(
            "--skip-tenant",
            action="store_true",
            help="Skip creating default tenant"
        )
        parser.add_argument(
            "--skip-superuser",
            action="store_true",
            help="Skip creating superuser"
        )
        parser.add_argument(
            "--tenant-only",
            action="store_true",
            help="Only create tenant and initialize apps (skip migrations and app loading)"
        )
        parser.add_argument(
            "--tenant-name",
            default="Default Organization",
            help="Name for the default tenant"
        )
        parser.add_argument(
            "--tenant-domain",
            default="default.local",
            help="Domain for the default tenant"
        )
        parser.add_argument(
            "--admin-username",
            default="admin",
            help="Username for the admin superuser"
        )
        parser.add_argument(
            "--admin-email",
            default="admin@default.local",
            help="Email for the admin superuser"
        )
        parser.add_argument(
            "--admin-password",
            help="Password for the admin superuser (auto-generated if not provided)"
        )
        parser.add_argument(
            "--no-input",
            action="store_true",
            help="Run in non-interactive mode with defaults"
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.HTTP_INFO("\n" + "=" * 60))
        self.stdout.write(self.style.HTTP_INFO(" Synara Development Environment Initialization"))
        self.stdout.write(self.style.HTTP_INFO(" BOOT-001 / PROD-BOOT-005 Compliant"))
        self.stdout.write(self.style.HTTP_INFO("=" * 60 + "\n"))

        # Track what we did
        results = {
            "migrations_applied": False,
            "apps_loaded": 0,
            "tenant_created": False,
            "tenant_id": None,
            "superuser_created": False,
            "superuser_username": None,
            "apps_installed": 0,
            "primitives_seeded": 0,
            "reflexes_seeded": 0,
        }

        try:
            # Step 1: Migrations
            if not options["tenant_only"] and not options["skip_migrate"]:
                results["migrations_applied"] = self._run_migrations()

            # Step 2: Load app definitions
            if not options["tenant_only"] and not options["skip_apps"]:
                results["apps_loaded"] = self._load_apps()

            # Step 3: Create tenant
            if not options["skip_tenant"]:
                tenant_info = self._ensure_tenant(
                    name=options["tenant_name"],
                    domain=options["tenant_domain"],
                    email=options["admin_email"],
                )
                results["tenant_created"] = tenant_info["created"]
                results["tenant_id"] = tenant_info["id"]

            # Step 4: Create superuser
            if not options["skip_superuser"] and results["tenant_id"]:
                user_info = self._ensure_superuser(
                    username=options["admin_username"],
                    email=options["admin_email"],
                    password=options["admin_password"],
                    tenant_id=results["tenant_id"],
                    no_input=options["no_input"],
                )
                results["superuser_created"] = user_info["created"]
                results["superuser_username"] = user_info["username"]
                results["password"] = user_info.get("password")

            # Step 5: Initialize apps for tenant
            if results["tenant_id"] and not options["skip_apps"]:
                results["apps_installed"] = self._initialize_apps(results["tenant_id"])

            # Step 6: Seed primitive registry (PRM-001 §8)
            if not options["tenant_only"]:
                results["primitives_seeded"] = self._seed_primitives()

            # Step 7: Seed reflexes (REF-001)
            if not options["tenant_only"]:
                results["reflexes_seeded"] = self._seed_reflexes()

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\nError during initialization: {e}"))
            raise CommandError(str(e))

        # Print summary
        self._print_summary(results)

    def _run_migrations(self):
        """Run database migrations."""
        self.stdout.write("\n[1/7] Running migrations...")

        try:
            call_command("migrate", "--run-syncdb", verbosity=0)
            self.stdout.write(self.style.SUCCESS("     Migrations applied successfully"))
            return True
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"     Migration warning: {e}"))
            return False

    def _load_apps(self):
        """Load app definitions from YAML files."""
        self.stdout.write("\n[2/7] Loading app definitions...")

        try:
            from syn.marketplace.app_registry import MarketplaceAppDefinition

            initial_count = MarketplaceAppDefinition.objects.count()
            call_command("load_apps", verbosity=0)
            final_count = MarketplaceAppDefinition.objects.count()
            new_apps = final_count - initial_count

            if new_apps > 0:
                self.stdout.write(self.style.SUCCESS(f"     Loaded {new_apps} new app definitions (total: {final_count})"))
            else:
                self.stdout.write(f"     No new apps to load (existing: {final_count})")

            return final_count
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"     App loading warning: {e}"))
            return 0

    @transaction.atomic
    def _ensure_tenant(self, name, domain, email):
        """Create default tenant if none exists."""
        self.stdout.write("\n[3/7] Ensuring tenant exists...")

        try:
            from syn.synara.models import Tenant
        except ImportError:
            from core.models import Tenant

        existing = Tenant.objects.first()
        if existing:
            self.stdout.write(f"     Tenant already exists: {existing.org_name} (ID: {existing.id})")
            return {"id": str(existing.id), "created": False}

        tenant = Tenant.objects.create(
            org_name=name,
            domain=domain,
            contact_email=email,
            is_active=True,
        )

        self.stdout.write(self.style.SUCCESS(f"     Created tenant: {tenant.org_name}"))
        self.stdout.write(f"     Tenant ID: {tenant.id}")

        return {"id": str(tenant.id), "created": True}

    @transaction.atomic
    def _ensure_superuser(self, username, email, password, tenant_id, no_input=False):
        """Create superuser and assign to tenant."""
        self.stdout.write("\n[4/7] Ensuring superuser exists...")

        # Check if user already exists
        existing = User.objects.filter(username=username).first()
        if existing:
            # Ensure user is assigned to tenant
            if str(getattr(existing, 'tenant_id', None)) != tenant_id:
                existing.tenant_id = tenant_id
                existing.save()
                self.stdout.write(f"     Assigned existing user '{username}' to tenant")
            else:
                self.stdout.write(f"     Superuser '{username}' already exists and assigned to tenant")
            return {"username": username, "created": False}

        # Generate password if not provided
        password_generated = password is None
        if password_generated:
            password = self._generate_password()

        # Create superuser
        user = User.objects.create_superuser(
            username=username,
            email=email,
            password=password,
        )

        # Assign to tenant
        user.tenant_id = tenant_id
        user.save()

        self.stdout.write(self.style.SUCCESS(f"     Created superuser: {username}"))
        self.stdout.write(f"     Email: {email}")

        result = {"username": username, "created": True}

        if password_generated:
            result["password"] = password

        return result

    def _initialize_apps(self, tenant_id):
        """Initialize apps for tenant."""
        self.stdout.write("\n[5/7] Initializing apps for tenant...")

        try:
            from syn.marketplace.app_registry import AppRegistryService

            result = AppRegistryService.initialize_tenant(tenant_id)
            licenses = result.get("licenses", 0)
            installs = result.get("installations", 0)

            self.stdout.write(self.style.SUCCESS(f"     Licensed {licenses} apps"))
            self.stdout.write(self.style.SUCCESS(f"     Installed {installs} apps"))

            return installs
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"     App initialization warning: {e}"))
            return 0

    def _seed_primitives(self):
        """Seed primitive registry (PRM-001 §8)."""
        self.stdout.write("\n[6/7] Seeding primitive registry (PRM-001 §8)...")

        try:
            from syn.primitives.models import PrimitiveRegistry

            initial_count = PrimitiveRegistry.objects.count()
            call_command("seed_primitive_registry", verbosity=0)
            final_count = PrimitiveRegistry.objects.count()
            new_primitives = final_count - initial_count

            if new_primitives > 0:
                self.stdout.write(self.style.SUCCESS(f"     Seeded {new_primitives} new primitives (total: {final_count})"))
            else:
                self.stdout.write(f"     No new primitives to seed (existing: {final_count})")

            return final_count
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"     Primitive seeding warning: {e}"))
            return 0

    def _seed_reflexes(self):
        """Seed reflexes from YAML definitions (REF-001)."""
        self.stdout.write("\n[7/7] Seeding reflexes (REF-001)...")

        try:
            from syn.reflex.models import Reflex

            initial_count = Reflex.objects.count()
            call_command("seed_reflexes", verbosity=0)
            final_count = Reflex.objects.count()
            new_reflexes = final_count - initial_count

            if new_reflexes > 0:
                self.stdout.write(self.style.SUCCESS(f"     Seeded {new_reflexes} new reflexes (total: {final_count})"))
            else:
                self.stdout.write(f"     No new reflexes to seed (existing: {final_count})")

            return final_count
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"     Reflex seeding warning: {e}"))
            return 0

    def _generate_password(self, length=12):
        """Generate a simple but valid password for dev environments."""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def _print_summary(self, results):
        """Print initialization summary."""
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS(" Initialization Complete"))
        self.stdout.write("=" * 60)

        self.stdout.write(f"\n  Migrations: {'Applied' if results['migrations_applied'] else 'Skipped'}")
        self.stdout.write(f"  App Definitions: {results['apps_loaded']} loaded")
        self.stdout.write(f"  Tenant: {'Created' if results['tenant_created'] else 'Existing'} (ID: {results['tenant_id']})")
        self.stdout.write(f"  Superuser: {'Created' if results['superuser_created'] else 'Existing'} ({results['superuser_username']})")
        self.stdout.write(f"  Installed Apps: {results['apps_installed']}")
        self.stdout.write(f"  Primitives: {results['primitives_seeded']} registered (PRM-001)")
        self.stdout.write(f"  Reflexes: {results['reflexes_seeded']} seeded (REF-001)")

        if results.get("password"):
            self.stdout.write(self.style.WARNING(f"\n  Generated Password: {results['password']}"))
            self.stdout.write(self.style.WARNING("  (Change this password after first login)"))

        self.stdout.write(self.style.SUCCESS(f"\n  Server ready! Start with: python manage.py runserver"))
        self.stdout.write(f"  Login at: http://localhost:8000/ui/login/")
        self.stdout.write("")

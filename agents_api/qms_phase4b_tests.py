"""Phase 4B tests — QMS-002 Resource Management (Employee, Commitment, ActionToken).

Standard: QMS-002 v1.0
CR: Phase 4B — Resource Management Models & Endpoints
"""

import json
import secrets
from datetime import date, timedelta

from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import User
from agents_api.models import (
    ActionToken,
    Employee,
    HoshinProject,
    ResourceCommitment,
    Site,
)
from core.models.project import Project
from core.models.tenant import Membership, Tenant

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_enterprise_user(username, email):
    """Create an enterprise-tier user with tenant + membership."""
    user = User.objects.create_user(username=username, email=email, password="test1234")
    user.tier = "enterprise"
    user.save(update_fields=["tier"])
    tenant = Tenant.objects.create(name=f"Tenant_{username}", slug=f"t-{username}")
    Membership.objects.create(user=user, tenant=tenant, role="owner")
    return user, tenant


def _make_hoshin_project(user, tenant, site, title="Test Project"):
    """Create a core.Project + HoshinProject wrapper."""
    core_proj = Project.objects.create(tenant=tenant, title=title)
    return HoshinProject.objects.create(project=core_proj, site=site)


# ============================================================================
# Model tests
# ============================================================================


class EmployeeModelTest(TestCase):
    """QMS-002 §2.1 — Employee model fields and constraints."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("emp_model", "emp_model@test.com")
        cls.site = Site.objects.create(
            tenant=cls.tenant,
            name="Plant A",
            code="FTW",
        )

    def test_employee_fields_exist(self):
        """Employee has all required fields per QMS-002 §2.1."""
        emp = Employee.objects.create(
            tenant=self.tenant,
            name="Alice",
            email="alice@test.com",
            role="Facilitator",
            department="Operations",
            site=self.site,
        )
        self.assertEqual(emp.name, "Alice")
        self.assertEqual(emp.email, "alice@test.com")
        self.assertEqual(emp.role, "Facilitator")
        self.assertEqual(emp.department, "Operations")
        self.assertEqual(emp.site, self.site)
        self.assertTrue(emp.is_active)

    def test_tenant_email_unique(self):
        """Email is unique per tenant."""
        Employee.objects.create(
            tenant=self.tenant,
            name="Bob",
            email="bob@test.com",
        )
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            Employee.objects.create(
                tenant=self.tenant,
                name="Bob2",
                email="bob@test.com",
            )

    def test_user_link_nullable(self):
        """user_link FK is optional (nullable)."""
        emp_no_link = Employee.objects.create(
            tenant=self.tenant,
            name="Carol",
            email="carol@test.com",
        )
        self.assertIsNone(emp_no_link.user_link)

        emp_linked = Employee.objects.create(
            tenant=self.tenant,
            name="Dave",
            email="dave@test.com",
            user_link=self.user,
        )
        self.assertEqual(emp_linked.user_link, self.user)

    def test_soft_delete(self):
        """Setting is_active=False soft-deletes the employee."""
        emp = Employee.objects.create(
            tenant=self.tenant,
            name="Eve",
            email="eve@test.com",
        )
        self.assertTrue(emp.is_active)
        emp.is_active = False
        emp.save(update_fields=["is_active"])
        emp.refresh_from_db()
        self.assertFalse(emp.is_active)


class ResourceCommitmentModelTest(TestCase):
    """QMS-002 §2.2 — ResourceCommitment fields, lifecycle, availability."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("rc_model", "rc_model@test.com")
        cls.site = Site.objects.create(tenant=cls.tenant, name="Plant B", code="DFW")
        cls.emp = Employee.objects.create(
            tenant=cls.tenant,
            name="Frank",
            email="frank@test.com",
        )
        cls.project = _make_hoshin_project(cls.user, cls.tenant, cls.site, "Reduce Waste")

    def test_commitment_fields_exist(self):
        """ResourceCommitment has all required fields per QMS-002 §2.2."""
        rc = ResourceCommitment.objects.create(
            employee=self.emp,
            project=self.project,
            role="facilitator",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 6, 30),
            requested_by=self.user,
        )
        self.assertEqual(rc.employee, self.emp)
        self.assertEqual(rc.project, self.project)
        self.assertEqual(rc.role, "facilitator")
        self.assertEqual(rc.start_date, date(2026, 4, 1))
        self.assertEqual(rc.end_date, date(2026, 6, 30))
        self.assertEqual(float(rc.hours_per_day), 8.0)
        self.assertEqual(rc.status, "requested")

    def test_role_choices(self):
        """Role field accepts all defined choices."""
        for role_val, _ in ResourceCommitment.ROLE_CHOICES:
            rc = ResourceCommitment.objects.create(
                employee=self.emp,
                project=self.project,
                role=role_val,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 31),
            )
            self.assertEqual(rc.role, role_val)

    def test_status_lifecycle_transitions(self):
        """Status lifecycle: requested → confirmed → active → completed."""
        rc = ResourceCommitment.objects.create(
            employee=self.emp,
            project=self.project,
            role="team_member",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 9, 30),
        )
        self.assertEqual(rc.status, "requested")

        # requested → confirmed
        valid = ResourceCommitment.VALID_TRANSITIONS["requested"]
        self.assertIn("confirmed", valid)
        rc.status = "confirmed"
        rc.save()

        # confirmed → active
        valid = ResourceCommitment.VALID_TRANSITIONS["confirmed"]
        self.assertIn("active", valid)
        rc.status = "active"
        rc.save()

        # active → completed
        valid = ResourceCommitment.VALID_TRANSITIONS["active"]
        self.assertIn("completed", valid)
        rc.status = "completed"
        rc.save()
        rc.refresh_from_db()
        self.assertEqual(rc.status, "completed")

    def test_availability_overlap_detection(self):
        """check_availability detects overlapping commitments."""
        ResourceCommitment.objects.create(
            employee=self.emp,
            project=self.project,
            role="facilitator",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 6, 30),
            status="confirmed",
        )
        # Overlapping range
        conflicts = ResourceCommitment.check_availability(
            self.emp,
            date(2026, 5, 1),
            date(2026, 7, 31),
        )
        self.assertEqual(conflicts.count(), 1)

        # Non-overlapping range
        no_conflicts = ResourceCommitment.check_availability(
            self.emp,
            date(2026, 7, 1),
            date(2026, 8, 31),
        )
        self.assertEqual(no_conflicts.count(), 0)


class ActionTokenModelTest(TestCase):
    """QMS-002 §2.3 — ActionToken security properties."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("at_model", "at_model@test.com")
        cls.emp = Employee.objects.create(
            tenant=cls.tenant,
            name="Grace",
            email="grace@test.com",
        )

    def test_auto_generated_token(self):
        """Token is auto-generated with ≥32 bytes of randomness."""
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="confirm_availability",
        )
        self.assertIsNotNone(tok.token)
        self.assertGreaterEqual(len(tok.token), 32)

    def test_default_72h_expiry(self):
        """Default expiry is ~72 hours from creation."""
        before = timezone.now()
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="decline",
        )
        after = timezone.now()
        expected_min = before + timedelta(hours=71, minutes=59)
        expected_max = after + timedelta(hours=72, minutes=1)
        self.assertGreaterEqual(tok.expires_at, expected_min)
        self.assertLessEqual(tok.expires_at, expected_max)

    def test_single_use(self):
        """use() sets used_at, making token invalid."""
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="update_progress",
        )
        self.assertTrue(tok.is_valid)
        self.assertIsNone(tok.used_at)

        tok.use()
        tok.refresh_from_db()
        self.assertIsNotNone(tok.used_at)
        self.assertFalse(tok.is_valid)

    def test_is_valid_expired(self):
        """Expired token is invalid."""
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="view_dashboard",
            expires_at=timezone.now() - timedelta(hours=1),
            token=secrets.token_urlsafe(32),
        )
        self.assertFalse(tok.is_valid)


# ============================================================================
# API tests
# ============================================================================


@SECURE_OFF
class EmployeeAPITest(TestCase):
    """QMS-002 §3.1 — Employee CRUD endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("emp_api", "emp_api@test.com")
        cls.site = Site.objects.create(tenant=cls.tenant, name="Plant C", code="AUS")

    def test_create_employee(self):
        """POST /api/hoshin/employees/ creates an employee."""
        self.client.force_login(self.user)
        resp = self.client.post(
            "/api/hoshin/employees/",
            data=json.dumps(
                {
                    "name": "Alice",
                    "email": "alice@example.com",
                    "role": "Engineer",
                    "department": "Quality",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["name"], "Alice")
        self.assertEqual(data["email"], "alice@example.com")

    def test_list_with_filter(self):
        """GET /api/hoshin/employees/?department=X filters correctly."""
        self.client.force_login(self.user)
        Employee.objects.create(
            tenant=self.tenant,
            name="Bob",
            email="bob@example.com",
            department="Quality",
        )
        Employee.objects.create(
            tenant=self.tenant,
            name="Carol",
            email="carol@example.com",
            department="Operations",
        )
        resp = self.client.get("/api/hoshin/employees/?department=Quality")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "Bob")

    def test_update_employee(self):
        """PUT /api/hoshin/employees/<id>/ updates fields."""
        emp = Employee.objects.create(
            tenant=self.tenant,
            name="Dave",
            email="dave@example.com",
        )
        self.client.force_login(self.user)
        resp = self.client.put(
            f"/api/hoshin/employees/{emp.id}/",
            data=json.dumps({"role": "Senior Engineer"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], "Senior Engineer")

    def test_soft_delete(self):
        """DELETE /api/hoshin/employees/<id>/ soft-deletes."""
        emp = Employee.objects.create(
            tenant=self.tenant,
            name="Eve",
            email="eve@example.com",
        )
        self.client.force_login(self.user)
        resp = self.client.delete(f"/api/hoshin/employees/{emp.id}/")
        self.assertEqual(resp.status_code, 200)
        emp.refresh_from_db()
        self.assertFalse(emp.is_active)


@SECURE_OFF
class CommitmentAPITest(TestCase):
    """QMS-002 §3.2 — ResourceCommitment CRUD endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("rc_api", "rc_api@test.com")
        cls.site = Site.objects.create(tenant=cls.tenant, name="Plant D", code="HOU")
        cls.emp = Employee.objects.create(
            tenant=cls.tenant,
            name="Frank",
            email="frank@example.com",
        )
        cls.project = _make_hoshin_project(cls.user, cls.tenant, cls.site, "Improve OEE")

    def test_create_commitment(self):
        """POST /api/hoshin/commitments/ creates a commitment."""
        self.client.force_login(self.user)
        resp = self.client.post(
            "/api/hoshin/commitments/",
            data=json.dumps(
                {
                    "employee_id": str(self.emp.id),
                    "project_id": str(self.project.id),
                    "role": "facilitator",
                    "start_date": "2026-04-01",
                    "end_date": "2026-06-30",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["role"], "facilitator")
        self.assertEqual(data["status"], "requested")

    def test_update_status_transition(self):
        """PUT /api/hoshin/commitments/<id>/ enforces valid transitions."""
        rc = ResourceCommitment.objects.create(
            employee=self.emp,
            project=self.project,
            role="team_member",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 6, 30),
            requested_by=self.user,
        )
        self.client.force_login(self.user)
        # requested → confirmed (valid)
        resp = self.client.put(
            f"/api/hoshin/commitments/{rc.id}/",
            data=json.dumps({"status": "confirmed"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "confirmed")

    def test_reject_invalid_transition(self):
        """PUT rejects invalid status transitions."""
        rc = ResourceCommitment.objects.create(
            employee=self.emp,
            project=self.project,
            role="team_member",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 9, 30),
            requested_by=self.user,
        )
        self.client.force_login(self.user)
        # requested → completed (invalid)
        resp = self.client.put(
            f"/api/hoshin/commitments/{rc.id}/",
            data=json.dumps({"status": "completed"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        # Middleware wraps errors — message is in "message" key
        self.assertIn("Cannot transition", body.get("message", str(body)))

    def test_list_by_project(self):
        """GET /api/hoshin/commitments/?project=<id> filters by project."""
        ResourceCommitment.objects.create(
            employee=self.emp,
            project=self.project,
            role="sponsor",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 6, 30),
        )
        self.client.force_login(self.user)
        resp = self.client.get(f"/api/hoshin/commitments/?project={self.project.id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)


@SECURE_OFF
class ActionTokenAPITest(TestCase):
    """QMS-002 §3.3 — ActionToken endpoint (no auth)."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("at_api", "at_api@test.com")
        cls.emp = Employee.objects.create(
            tenant=cls.tenant,
            name="Grace",
            email="grace@example.com",
        )

    def test_get_returns_scoped_info(self):
        """GET /action/<token>/ returns token info."""
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="confirm_availability",
            scoped_to={"project_id": "abc-123"},
        )
        resp = self.client.get(f"/action/{tok.token}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["action_type"], "confirm_availability")
        self.assertTrue(data["is_valid"])

    def test_post_marks_used(self):
        """POST /action/<token>/ executes action and marks token used."""
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="decline",
        )
        resp = self.client.post(f"/action/{tok.token}/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        # Token is now used — second POST returns 410
        resp2 = self.client.post(f"/action/{tok.token}/")
        self.assertEqual(resp2.status_code, 410)

    def test_expired_token_returns_410(self):
        """Expired token returns 410 Gone."""
        tok = ActionToken.objects.create(
            employee=self.emp,
            action_type="update_progress",
            expires_at=timezone.now() - timedelta(hours=1),
            token=secrets.token_urlsafe(32),
        )
        resp = self.client.get(f"/action/{tok.token}/")
        self.assertEqual(resp.status_code, 410)


# ============================================================================
# Phase 4C — Bulk Import, Timeline, Facilitator Calendar
# ============================================================================


@SECURE_OFF
class EmployeeImportTest(TestCase):
    """QMS-002 §3.1 — Bulk CSV import of employees."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("imp_api", "imp_api@test.com")
        cls.site = Site.objects.create(tenant=cls.tenant, name="Plant Import", code="IMP")

    def _make_csv(self, rows):
        """Build an in-memory CSV file from list of dicts."""
        import io

        lines = []
        if rows:
            lines.append(",".join(rows[0].keys()))
            for r in rows:
                lines.append(",".join(str(v) for v in r.values()))
        content = "\n".join(lines).encode("utf-8")
        f = io.BytesIO(content)
        f.name = "employees.csv"
        return f

    def test_import_creates_employees(self):
        """CSV upload creates new employees."""
        self.client.force_login(self.user)
        csv_file = self._make_csv(
            [
                {
                    "name": "Alice",
                    "email": "alice@import.com",
                    "role": "Engineer",
                    "department": "QA",
                },
                {
                    "name": "Bob",
                    "email": "bob@import.com",
                    "role": "Technician",
                    "department": "Ops",
                },
            ]
        )
        resp = self.client.post(
            "/api/hoshin/employees/import/",
            {"file": csv_file},
            format="multipart",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["created"], 2)
        self.assertEqual(data["updated"], 0)
        self.assertTrue(Employee.objects.filter(tenant=self.tenant, email="alice@import.com").exists())

    def test_import_deduplicates_by_email(self):
        """Re-importing same email updates instead of creating duplicate."""
        Employee.objects.create(
            tenant=self.tenant,
            name="Carol",
            email="carol@import.com",
            role="Old Role",
        )
        self.client.force_login(self.user)
        csv_file = self._make_csv(
            [
                {
                    "name": "Carol Updated",
                    "email": "carol@import.com",
                    "role": "New Role",
                    "department": "Engineering",
                },
            ]
        )
        resp = self.client.post(
            "/api/hoshin/employees/import/",
            {"file": csv_file},
            format="multipart",
        )
        data = resp.json()
        self.assertEqual(data["created"], 0)
        self.assertEqual(data["updated"], 1)
        carol = Employee.objects.get(tenant=self.tenant, email="carol@import.com")
        self.assertEqual(carol.name, "Carol Updated")
        self.assertEqual(carol.role, "New Role")

    def test_import_missing_columns_returns_error(self):
        """CSV without required columns returns 400."""
        self.client.force_login(self.user)
        csv_file = self._make_csv(
            [
                {"foo": "bar", "baz": "qux"},
            ]
        )
        resp = self.client.post(
            "/api/hoshin/employees/import/",
            {"file": csv_file},
            format="multipart",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("name", body.get("message", str(body)))


@SECURE_OFF
class EmployeeTimelineTest(TestCase):
    """QMS-002 §3.2 — Employee timeline endpoint."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("tl_api", "tl_api@test.com")
        cls.site = Site.objects.create(tenant=cls.tenant, name="Plant TL", code="TL1")
        cls.emp = Employee.objects.create(
            tenant=cls.tenant,
            name="Dave",
            email="dave@tl.com",
        )
        cls.project = _make_hoshin_project(cls.user, cls.tenant, cls.site, "Timeline Project")
        # Active commitment in April-June 2026
        ResourceCommitment.objects.create(
            employee=cls.emp,
            project=cls.project,
            role="facilitator",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 6, 30),
            status="active",
            hours_per_day=4,
        )
        # Completed commitment (should be excluded)
        ResourceCommitment.objects.create(
            employee=cls.emp,
            project=cls.project,
            role="team_member",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 3, 31),
            status="completed",
            hours_per_day=8,
        )

    def test_timeline_returns_active_commitments(self):
        """Timeline returns active commitments within the date range."""
        self.client.force_login(self.user)
        resp = self.client.get(
            f"/api/hoshin/employees/{self.emp.id}/timeline/?start=2026-01-01&end=2026-12-31",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["commitments"]), 1)  # Only the active one
        self.assertEqual(data["employee"]["name"], "Dave")

    def test_timeline_excludes_completed(self):
        """Completed and declined commitments are excluded."""
        self.client.force_login(self.user)
        resp = self.client.get(
            f"/api/hoshin/employees/{self.emp.id}/timeline/?start=2026-01-01&end=2026-03-31",
        )
        data = resp.json()
        self.assertEqual(len(data["commitments"]), 0)  # Completed one excluded


@SECURE_OFF
class FacilitatorCalendarTest(TestCase):
    """QMS-002 §3.4 — Facilitator workload with over-commitment detection."""

    @classmethod
    def setUpTestData(cls):
        cls.user, cls.tenant = _make_enterprise_user("fac_api", "fac_api@test.com")
        cls.site = Site.objects.create(tenant=cls.tenant, name="Plant FC", code="FC1")
        cls.emp = Employee.objects.create(
            tenant=cls.tenant,
            name="Eve",
            email="eve@fc.com",
        )
        cls.proj1 = _make_hoshin_project(cls.user, cls.tenant, cls.site, "Project Alpha")
        cls.proj2 = _make_hoshin_project(cls.user, cls.tenant, cls.site, "Project Beta")

        # Two overlapping facilitator commitments (5h + 5h = 10h > 8h)
        ResourceCommitment.objects.create(
            employee=cls.emp,
            project=cls.proj1,
            role="facilitator",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 10),
            status="confirmed",
            hours_per_day=5,
        )
        ResourceCommitment.objects.create(
            employee=cls.emp,
            project=cls.proj2,
            role="facilitator",
            start_date=date(2026, 4, 5),
            end_date=date(2026, 4, 15),
            status="confirmed",
            hours_per_day=5,
        )

    def test_returns_facilitators(self):
        """Facilitator calendar returns facilitators with commitments."""
        self.client.force_login(self.user)
        resp = self.client.get("/api/hoshin/calendar/facilitators/?fiscal_year=2026")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["facilitators"]), 1)
        self.assertEqual(data["facilitators"][0]["employee"]["name"], "Eve")
        self.assertEqual(len(data["facilitators"][0]["commitments"]), 2)

    def test_detects_over_commitment(self):
        """Over-commitment detected when hours > 8 on overlapping days."""
        self.client.force_login(self.user)
        resp = self.client.get("/api/hoshin/calendar/facilitators/?fiscal_year=2026")
        data = resp.json()
        fac = data["facilitators"][0]
        self.assertTrue(fac["over_committed"])
        self.assertGreater(fac["over_committed_days"], 0)

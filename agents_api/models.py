"""Agents API models."""

import secrets
import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

# ---------------------------------------------------------------------------
# Re-exports: models transferred to extracted apps (backward compatibility)
# These will be removed once all imports are updated to point at new apps.
# ---------------------------------------------------------------------------
from a3.models import A3Report  # noqa: F401
from ce_matrix.models import CEMatrix  # noqa: F401
from dsw.models import DSWResult, SavedModel  # noqa: F401
from fmea.models import FMEA, FMEARow  # noqa: F401
from hoshin.models import (  # noqa: F401
    ActionItem,
    ActionToken,
    AnnualObjective,
    HoshinKPI,
    HoshinProject,
    ProjectTemplate,
    ResourceCommitment,
    StrategicObjective,
    XMatrixCorrelation,
)
from ishikawa.models import IshikawaDiagram  # noqa: F401
from learn.models import AssessmentAttempt, LearnSession, SectionProgress  # noqa: F401
from plantsim.models import PlantSimulation  # noqa: F401
from rack.models import RackSession  # noqa: F401
from rca.models import RCASession  # noqa: F401
from reports.models import Report  # noqa: F401
from syn.core.base_models import SynaraImmutableLog
from triage.models import AgentLog, TriageResult  # noqa: F401


class CacheEntry(models.Model):
    """Database-backed cache entry with TTL support.

    Stores serialized Python objects with expiration times.
    Tempora cleans up expired entries via scheduled task.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    key = models.CharField(max_length=255, unique=True, db_index=True)
    value = models.BinaryField()  # Pickled Python object
    value_type = models.CharField(max_length=50, default="pickle")  # pickle, json

    # TTL support
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # Metadata
    namespace = models.CharField(max_length=50, blank=True, db_index=True)
    user_id = models.IntegerField(null=True, blank=True, db_index=True)
    hit_count = models.IntegerField(default=0)
    last_accessed = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "session_cache"
        indexes = [
            models.Index(fields=["namespace", "key"]),
            models.Index(fields=["expires_at"]),
            models.Index(fields=["user_id", "namespace"]),
        ]

    def __str__(self):
        return f"{self.key} (expires: {self.expires_at})"

    @property
    def is_expired(self) -> bool:
        from django.utils import timezone

        if self.expires_at is None:
            return False
        return timezone.now() > self.expires_at


# =============================================================================
# LLM Usage Tracking & Rate Limiting — moved to llm/ app
# =============================================================================
from llm.models import (
    LLM_RATE_LIMITS,  # noqa: F401
    LLMUsage,  # noqa: F401
    RateLimitOverride,  # noqa: F401
    check_rate_limit,  # noqa: F401
)

# =============================================================================
# Whiteboard - Collaborative Boards
# =============================================================================


def generate_room_code():
    """Generate a 6-character room code."""
    import random
    import string

    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


class Board(models.Model):
    """Collaborative whiteboard for kaizen sessions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    room_code = models.CharField(max_length=10, unique=True, default=generate_room_code, db_index=True)
    name = models.CharField(max_length=255, default="Untitled Board")

    # Owner/creator
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_boards",
    )

    # Optional link to a project (for kaizen sessions tied to investigations)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="boards",
    )

    # Board state as JSON
    elements = models.JSONField(default=list)  # List of element objects
    connections = models.JSONField(default=list)  # List of connection objects

    # Canvas state
    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    # Voting state
    is_voting_active = models.BooleanField(default=False, db_column="voting_active")
    votes_per_user = models.IntegerField(default=3)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Version for conflict detection (incremented on each save)
    version = models.IntegerField(default=0)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.room_code})"

    def save(self, *args, **kwargs):
        from django.db.models import F

        is_new = self._state.adding
        super().save(*args, **kwargs)
        if not is_new:
            # Atomic version increment — single query, no race window
            type(self).objects.filter(pk=self.pk).update(version=F("version") + 1)
            self.refresh_from_db(fields=["version"])


class BoardParticipant(models.Model):
    """Track who's in a board session."""

    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="participants")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="board_participations",
    )

    # Participant color for cursors/attribution
    color = models.CharField(max_length=7, default="#4a9f6e")  # Hex color

    # Last activity (for presence detection)
    last_seen = models.DateTimeField(auto_now=True)

    # Cursor position (for showing where others are)
    cursor_x = models.FloatField(null=True, blank=True)
    cursor_y = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("board", "user")

    def __str__(self):
        return f"{self.user.username} in {self.board.room_code}"


class BoardVote(models.Model):
    """Dot vote on a board element."""

    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="votes")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="board_votes",
    )
    guest_invite = models.ForeignKey(
        "BoardGuestInvite",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="votes",
    )
    element_id = models.CharField(max_length=50)  # The element ID being voted on
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["board", "user", "element_id"],
                condition=models.Q(user__isnull=False),
                name="unique_user_vote",
            ),
            models.UniqueConstraint(
                fields=["board", "guest_invite", "element_id"],
                condition=models.Q(guest_invite__isnull=False),
                name="unique_guest_vote",
            ),
        ]

    def __str__(self):
        if self.user:
            return f"{self.user.username} voted on {self.element_id}"
        return f"Guest voted on {self.element_id}"


class BoardGuestInvite(models.Model):
    """Guest invite for board access without a Svend account.

    Guests get scoped access to a single board via a unique token URL.
    No navigation, no access to other tools. Owner controls permission level.
    """

    class Permission(models.TextChoices):
        VIEW = "view", "View Only"
        EDIT = "edit", "Edit"
        EDIT_VOTE = "edit_vote", "Edit + Vote"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="guest_invites")
    token = models.CharField(max_length=64, unique=True, db_index=True)
    display_name = models.CharField(max_length=100, blank=True)
    permission = models.CharField(
        max_length=10,
        choices=Permission.choices,
        default=Permission.VIEW,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    color = models.CharField(max_length=7, default="#ff7eb9")
    last_seen = models.DateTimeField(null=True, blank=True)
    cursor_x = models.FloatField(null=True, blank=True)
    cursor_y = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["board", "is_active"]),
        ]

    def __str__(self):
        name = self.display_name or "(unnamed)"
        return f"Guest '{name}' on {self.board.room_code} ({self.permission})"

    @property
    def is_expired(self):
        from django.utils import timezone

        return timezone.now() > self.expires_at

    @property
    def is_valid(self):
        return self.is_active and not self.is_expired

    @classmethod
    def generate_token(cls):

        return secrets.token_hex(32)


class ValueStreamMap(models.Model):
    """Value Stream Map for lean process analysis.

    VSM visualizes material and information flow to identify waste.
    Unlike a general whiteboard, VSM has structured elements:
    - Process steps with cycle time, wait time, etc.
    - Inventory triangles between steps
    - Information flow (customer demand, scheduling)
    - Timeline showing value-add vs non-value-add time
    """

    class Status(models.TextChoices):
        CURRENT = "current", "Current State"
        FUTURE = "future", "Future State"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="value_stream_maps",
        null=True,
        blank=True,
    )

    # Optional link to project
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="value_stream_maps",
    )

    # Header
    name = models.CharField(max_length=255, default="Untitled VSM")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.CURRENT)
    fiscal_year = models.CharField(
        max_length=10,
        blank=True,
        default="",
        help_text="Fiscal year scope, e.g. '2026'. Empty = unscoped.",
    )
    paired_with = models.OneToOneField(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="paired_map",
        help_text="Current <-> Future state pairing",
    )
    product_family = models.CharField(max_length=255, blank=True, help_text="Product or service being mapped")

    # Customer info (right side of VSM)
    customer_name = models.CharField(max_length=255, blank=True, default="Customer")
    customer_demand = models.CharField(max_length=100, blank=True, help_text="e.g., 460 units/day")
    takt_time = models.FloatField(null=True, blank=True, help_text="Takt time in seconds")

    # Supplier info (left side of VSM)
    supplier_name = models.CharField(max_length=255, blank=True, default="Supplier")
    supply_frequency = models.CharField(max_length=100, blank=True, help_text="e.g., Weekly")

    # Multiple customers/suppliers as JSON arrays
    # Each: {id, name, detail, x, y}
    customers = models.JSONField(default=list, help_text="Customer entities on the map")
    suppliers = models.JSONField(default=list, help_text="Supplier entities on the map")

    # Process steps as JSON
    # Each step: {id, name, x, y, cycle_time, changeover_time, uptime, operators, shifts, batch_size, ...}
    process_steps = models.JSONField(default=list, help_text="Process boxes with metrics")

    # Inventory between steps
    # Each: {id, before_step_id, quantity, days_of_supply, x, y}
    inventory = models.JSONField(default=list, help_text="Inventory triangles")

    # Information flow
    # Each: {id, from_id, to_id, type: 'electronic'|'manual'|'schedule', label}
    information_flow = models.JSONField(default=list, help_text="Information arrows")

    # Material flow connections
    # Each: {id, from_step_id, to_step_id, type: 'push'|'pull'|'fifo'}
    material_flow = models.JSONField(default=list, help_text="Material flow arrows")

    # Timeline summary (calculated)
    total_lead_time = models.FloatField(null=True, blank=True, help_text="Total lead time in days")
    total_process_time = models.FloatField(null=True, blank=True, help_text="Total value-add time in seconds")
    pce = models.FloatField(null=True, blank=True, help_text="Process Cycle Efficiency (%)")

    # Kaizen bursts (improvement opportunities)
    # Each: {id, x, y, text, priority}
    kaizen_bursts = models.JSONField(default=list, help_text="Improvement opportunities")

    # Work centers (grouped parallel machines)
    # Each: {id, name, x, y, width, height}
    # Process steps link via optional work_center_id field
    work_centers = models.JSONField(default=list, help_text="Work center groupings")

    # Metric history for timeline tracking
    metric_snapshots = models.JSONField(
        default=list,
        blank=True,
        help_text="[{timestamp, lead_time, process_time, pce, takt_time, step_count, inventory_count}]",
    )

    # Canvas state
    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "value_stream_maps"
        ordering = ["-updated_at"]
        verbose_name = "Value Stream Map"
        verbose_name_plural = "Value Stream Maps"

    def __str__(self):
        return f"VSM: {self.name} ({self.status})"

    def calculate_metrics(self):
        """Calculate total lead time, process time, and PCE.

        For steps in work centers (parallel machines), uses effective cycle time:
        effective_CT = 1 / sum(1/CT_i) for all machines in the work center.

        PCE = value-add time / total lead time. Changeover time is NVA and
        included in total lead time but NOT in value-add process time.
        """
        total_ct = 0.0  # Value-add cycle time in seconds
        total_changeover = 0.0  # Changeover/setup time in seconds (NVA)
        total_wait = 0.0  # Wait time in days

        # Group steps by work center
        wc_steps = {}  # work_center_id -> [cycle_times]
        standalone_cts = []

        for step in self.process_steps:
            ct = step.get("cycle_time", 0) or 0
            co = step.get("changeover_time", 0) or 0
            total_changeover += co
            wc_id = step.get("work_center_id")
            if wc_id:
                wc_steps.setdefault(wc_id, []).append(ct)
            else:
                standalone_cts.append(ct)

        # Standalone steps: sum cycle times directly
        total_ct += sum(standalone_cts)

        # Work center steps: effective CT = 1 / sum(1/CT_i)
        for wc_id, cts in wc_steps.items():
            rate_sum = sum(1.0 / ct for ct in cts if ct > 0)
            if rate_sum > 0:
                total_ct += 1.0 / rate_sum

        for inv in self.inventory:
            days = inv.get("days_of_supply", 0) or 0
            total_wait += days

        self.total_process_time = total_ct
        # Total lead time includes changeover (NVA) in addition to process time and wait
        total_active_seconds = total_ct + total_changeover
        self.total_lead_time = total_wait + (total_active_seconds / 86400)

        if self.total_lead_time > 0:
            # PCE = value-add time / total lead time (changeover is NOT value-add)
            self.pce = round((total_ct / 86400 / self.total_lead_time) * 100, 4)
        else:
            self.pce = 0

        # Append metric snapshot if values changed
        from datetime import datetime, timezone

        snap = {
            "lead_time": round(self.total_lead_time or 0, 4),
            "process_time": round(self.total_process_time or 0, 1),
            "pce": round(self.pce or 0, 2),
            "takt_time": self.takt_time,
            "step_count": len(self.process_steps or []),
            "inventory_count": len(self.inventory or []),
        }
        snapshots = self.metric_snapshots or []
        last = snapshots[-1] if snapshots else None
        changed = (
            not last
            or last.get("lead_time") != snap["lead_time"]
            or last.get("process_time") != snap["process_time"]
            or last.get("pce") != snap["pce"]
            or last.get("takt_time") != snap["takt_time"]
        )
        if changed and (snap["step_count"] > 0 or snap["inventory_count"] > 0):
            snap["timestamp"] = datetime.now(timezone.utc).isoformat()
            snapshots.append(snap)
            if len(snapshots) > 100:
                snapshots = snapshots[-100:]
            self.metric_snapshots = snapshots

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "name": self.name,
            "status": self.status,
            "fiscal_year": self.fiscal_year,
            "paired_with_id": str(self.paired_with_id) if self.paired_with_id else None,
            "product_family": self.product_family,
            "customer_name": self.customer_name,
            "customer_demand": self.customer_demand,
            "takt_time": self.takt_time,
            "supplier_name": self.supplier_name,
            "supply_frequency": self.supply_frequency,
            "customers": self.customers,
            "suppliers": self.suppliers,
            "process_steps": self.process_steps,
            "inventory": self.inventory,
            "information_flow": self.information_flow,
            "material_flow": self.material_flow,
            "total_lead_time": self.total_lead_time,
            "total_process_time": self.total_process_time,
            "pce": self.pce,
            "kaizen_bursts": self.kaizen_bursts,
            "work_centers": self.work_centers,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "metric_snapshots": self.metric_snapshots or [],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Site(models.Model):
    """Manufacturing site within an enterprise tenant.

    One Tenant (billing entity) can have many Sites (physical plants).
    Contacts are text fields — operators/managers often don't have Svend accounts.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="sites",
    )
    name = models.CharField(max_length=255)
    code = models.CharField(
        max_length=20,
        blank=True,
        help_text="Short site code, e.g. PLT-01",
    )
    business_unit = models.CharField(max_length=255, blank=True)
    plant_manager = models.CharField(max_length=255, blank=True)
    ci_leader = models.CharField(max_length=255, blank=True)
    controller = models.CharField(max_length=255, blank=True)
    address = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_sites"
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.code})" if self.code else self.name

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "code": self.code,
            "business_unit": self.business_unit,
            "plant_manager": self.plant_manager,
            "ci_leader": self.ci_leader,
            "controller": self.controller,
            "address": self.address,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SiteAccess(models.Model):
    """Per-site access control for Hoshin Kanri.

    Org owners/admins bypass this — they see all sites.
    Members/viewers need an explicit SiteAccess entry to see a site's projects.
    """

    class SiteRole(models.TextChoices):
        VIEWER = "viewer", "Viewer"
        MEMBER = "member", "Member"
        ADMIN = "admin", "Admin"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.ForeignKey(
        Site,
        on_delete=models.CASCADE,
        related_name="access_list",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="site_access",
    )
    role = models.CharField(
        max_length=20,
        choices=SiteRole.choices,
        default=SiteRole.MEMBER,
    )
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "hoshin_site_access"
        unique_together = [["site", "user"]]

    def __str__(self):
        return f"{self.user} → {self.site} ({self.role})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "site_id": str(self.site_id),
            "user_id": self.user_id,
            "username": self.user.username,
            "display_name": getattr(self.user, "display_name", "") or self.user.username,
            "email": self.user.email,
            "role": self.role,
            "granted_by_id": self.granted_by_id,
            "created_at": self.created_at.isoformat(),
        }


class Employee(models.Model):
    """CI participant / contact within a tenant.

    Employees are non-user personnel records. A facilitator with a Svend
    account has ``user_link`` set; a team member on the plant floor who
    participates via email does not.  One record per person per tenant,
    deduplicated by email.

    Standard: QMS-002 §2.1
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="employees",
    )
    name = models.CharField(max_length=255)
    email = models.EmailField()
    role = models.CharField(max_length=255, blank=True)
    department = models.CharField(max_length=255, blank=True)
    site = models.ForeignKey(
        Site,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="employees",
    )
    user_link = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="employee_profile",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_employees"
        unique_together = [("tenant", "email")]

    def __str__(self):
        return f"{self.name} ({self.email})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "department": self.department,
            "site_id": str(self.site_id) if self.site_id else None,
            "user_link_id": str(self.user_link_id) if self.user_link_id else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class NonconformanceRecord(models.Model):
    """NCR tracker per ISO 9001 clause 10.2."""

    class Severity(models.TextChoices):
        MINOR = "minor", "Minor"
        MAJOR = "major", "Major"
        CRITICAL = "critical", "Critical"

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        INVESTIGATION = "investigation", "Investigation"
        CAPA = "capa", "CAPA"
        VERIFICATION = "verification", "Verification"
        CLOSED = "closed", "Closed"

    class Source(models.TextChoices):
        INTERNAL_AUDIT = "internal_audit", "Internal Audit"
        CUSTOMER_COMPLAINT = "customer_complaint", "Customer Complaint"
        SUPPLIER = "supplier", "Supplier"
        PROCESS = "process", "Process"
        EXTERNAL_AUDIT = "external_audit", "External Audit"
        MANAGEMENT_REVIEW = "management_review", "Management Review"
        OTHER = "other", "Other"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ncrs",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ncr_records",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="ncrs_created",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    severity = models.CharField(max_length=20, choices=Severity.choices, default=Severity.MINOR)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.OPEN)
    source = models.CharField(max_length=30, choices=Source.choices, default=Source.OTHER)
    iso_clause = models.CharField(max_length=20, blank=True, help_text="e.g. 8.7, 10.2")
    containment_action = models.TextField(blank=True)
    root_cause = models.TextField(blank=True)
    corrective_action = models.TextField(blank=True)
    verification_result = models.TextField(blank=True)
    capa_due_date = models.DateField(null=True, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional project link
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ncrs",
    )

    # Workflow fields
    raised_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="raised_ncrs",
    )
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_ncrs",
    )
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_ncrs",
    )

    # Cross-tool links
    rca_session = models.ForeignKey(
        "rca.RCASession",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ncrs",
    )
    capa_report = models.ForeignKey(
        "reports.Report",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ncrs",
    )

    # Supplier link (when source=supplier)
    supplier = models.ForeignKey(
        "agents_api.SupplierRecord",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ncrs",
    )

    # Graph linkage (GRAPH-001) — which process graph nodes this NCR relates to
    linked_process_node_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="UUIDs of ProcessNode records this NCR relates to — opt-in, not required",
    )

    # Evidence attachments
    files = models.ManyToManyField(
        "files.UserFile",
        blank=True,
        related_name="ncrs",
    )

    # Valid status transitions (forward and backward)
    TRANSITIONS = {
        "open": {"investigation"},
        "investigation": {"open", "capa"},
        "capa": {"investigation", "verification"},
        "verification": {"capa", "closed"},
        "closed": {"verification"},
    }
    # Requirements per transition target
    TRANSITION_REQUIRES = {
        "investigation": ["assigned_to"],
        "capa": ["root_cause"],
        "verification": ["_corrective_action_or_capa"],
        "closed": ["approved_by"],
    }

    class Meta:
        db_table = "iso_ncrs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"NCR: {self.title} ({self.severity})"

    def can_transition(self, new_status):
        """Check if transition is valid and requirements are met."""
        allowed = self.TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            return False, f"Cannot transition from '{self.status}' to '{new_status}'"
        for field in self.TRANSITION_REQUIRES.get(new_status, []):
            # A1: ISO 9001 §10.2.1(b) — verification requires documented corrective action
            if field == "_corrective_action_or_capa":
                has_ca = self.corrective_action and self.corrective_action.strip()
                has_capa = self.capa_report_id is not None
                if not has_ca and not has_capa:
                    return (
                        False,
                        "Corrective action or linked CAPA report is required before verification",
                    )
                continue
            # Handle both text fields and FK fields safely (BUG-05)
            val = getattr(self, field, None)
            if val is None:
                return False, f"'{field}' is required to transition to '{new_status}'"
            if isinstance(val, str) and not val.strip():
                return False, f"'{field}' is required to transition to '{new_status}'"
        return True, ""

    def to_dict(self):
        d = {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "status": self.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "source": self.source,
            "supplier_id": str(self.supplier_id) if self.supplier_id else None,
            "supplier_name": (self.supplier.name if self.supplier_id and self.supplier else None),
            "iso_clause": self.iso_clause,
            "containment_action": self.containment_action,
            "root_cause": self.root_cause,
            "corrective_action": self.corrective_action,
            "verification_result": self.verification_result,
            "capa_due_date": str(self.capa_due_date) if self.capa_due_date else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "raised_by": None,
            "assigned_to": None,
            "approved_by": None,
            "project_id": str(self.project_id) if self.project_id else None,
            "rca_session_id": str(self.rca_session_id) if self.rca_session_id else None,
            "capa_report_id": str(self.capa_report_id) if self.capa_report_id else None,
            "file_ids": [str(f.id) for f in self.files.all()],
            "status_changes": [],
        }
        if self.raised_by:
            d["raised_by"] = {
                "id": self.raised_by_id,
                "name": self.raised_by.display_name or self.raised_by.email,
            }
        if self.assigned_to:
            d["assigned_to"] = {
                "id": self.assigned_to_id,
                "name": self.assigned_to.display_name or self.assigned_to.email,
            }
        if self.approved_by:
            d["approved_by"] = {
                "id": self.approved_by_id,
                "name": self.approved_by.display_name or self.approved_by.email,
            }
        try:
            d["status_changes"] = [sc.to_dict() for sc in self.status_changes.order_by("created_at")]
        except Exception:
            pass
        try:
            d["field_changes"] = [
                fc.to_dict()
                for fc in QMSFieldChange.objects.filter(record_type="ncr", record_id=self.id).select_related(
                    "changed_by"
                )[:50]
            ]
        except Exception:
            d["field_changes"] = []
        return d


class NCRStatusChange(models.Model):
    """Status change history for NCRs."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ncr = models.ForeignKey(
        NonconformanceRecord,
        on_delete=models.CASCADE,
        related_name="status_changes",
    )
    from_status = models.CharField(max_length=20)
    to_status = models.CharField(max_length=20)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_ncr_status_changes"
        ordering = ["created_at"]

    def to_dict(self):
        return {
            "id": str(self.id),
            "from_status": self.from_status,
            "to_status": self.to_status,
            "changed_by": ((self.changed_by.display_name or self.changed_by.email) if self.changed_by else None),
            "note": self.note,
            "created_at": self.created_at.isoformat(),
        }


class CAPAReport(models.Model):
    """Corrective and Preventive Action per ISO 9001 §10.2 / FDA 21 CFR 820.90.

    Standalone model extracted from generic Report. Tracks the full CAPA lifecycle
    from containment through verification, with source linking to NCRs, audit
    findings, SPC alarms, and other QMS triggers.
    """

    class SourceType(models.TextChoices):
        NCR = "ncr", "Nonconformance Record"
        AUDIT_FINDING = "audit_finding", "Audit Finding"
        CUSTOMER_COMPLAINT = "customer_complaint", "Customer Complaint"
        SPC_ALARM = "spc_alarm", "SPC Alarm"
        MANAGEMENT_REVIEW = "management_review", "Management Review"
        SUPPLIER_ISSUE = "supplier_issue", "Supplier Issue"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        CONTAINMENT = "containment", "Containment"
        INVESTIGATION = "investigation", "Investigation"
        CORRECTIVE = "corrective", "Corrective Action"
        VERIFICATION = "verification", "Verification"
        CLOSED = "closed", "Closed"

    class Priority(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"

    TRANSITIONS = {
        "draft": {"containment"},
        "containment": {"draft", "investigation"},
        "investigation": {"containment", "corrective"},
        "corrective": {"investigation", "verification"},
        "verification": {"corrective", "closed"},
        "closed": {"verification"},
    }

    TRANSITION_REQUIRES = {
        "investigation": ["containment_action"],
        "corrective": ["root_cause"],
        "verification": ["corrective_action", "preventive_action"],
        "closed": ["verification_result"],
    }

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="capa_reports",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="capa_records",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="capa_reports_created",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="capa_reports",
    )

    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    priority = models.CharField(max_length=20, choices=Priority.choices, default=Priority.MEDIUM)

    source_type = models.CharField(max_length=30, choices=SourceType.choices, blank=True)
    source_id = models.UUIDField(null=True, blank=True, help_text="UUID of the source record")

    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_capas",
    )
    due_date = models.DateField(null=True, blank=True)

    containment_action = models.TextField(blank=True)
    root_cause = models.TextField(blank=True)
    corrective_action = models.TextField(blank=True)
    preventive_action = models.TextField(blank=True)
    verification_method = models.TextField(blank=True)
    verification_result = models.TextField(blank=True)

    effectiveness_check_date = models.DateField(null=True, blank=True)
    is_recurrence_checked = models.BooleanField(default=False, db_column="recurrence_check")
    cost_of_poor_quality = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
    )
    copq_category = models.CharField(
        max_length=30,
        blank=True,
        choices=[
            ("scrap", "Scrap"),
            ("rework", "Rework"),
            ("warranty", "Warranty"),
            ("inspection", "Inspection"),
            ("sorting", "Sorting"),
            ("other", "Other"),
        ],
        help_text="Direct cost category",
    )
    copq_paf_class = models.CharField(
        max_length=20,
        blank=True,
        choices=[
            ("prevention", "Prevention"),
            ("appraisal", "Appraisal"),
            ("internal_failure", "Internal Failure"),
            ("external_failure", "External Failure"),
        ],
        help_text="PAF model classification",
    )

    rca_session = models.ForeignKey(
        "rca.RCASession",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="capa_reports",
    )

    closed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_capas"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "status"]),
            models.Index(fields=["source_type", "source_id"]),
        ]
        verbose_name = "CAPA Report"

    def __str__(self):
        return f"CAPA: {self.title} ({self.priority})"

    def can_transition(self, new_status):
        """Check if transition is valid and requirements are met."""
        allowed = self.TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            return False, f"Cannot transition from '{self.status}' to '{new_status}'"
        for field in self.TRANSITION_REQUIRES.get(new_status, []):
            # Handle both text fields and FK fields safely (BUG-05)
            val = getattr(self, field, None)
            if val is None:
                return False, f"'{field}' is required to transition to '{new_status}'"
            if isinstance(val, str) and not val.strip():
                return False, f"'{field}' is required to transition to '{new_status}'"
        return True, ""

    def to_dict(self):
        d = {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "priority": self.priority,
            "source_type": self.source_type,
            "source_id": str(self.source_id) if self.source_id else None,
            "containment_action": self.containment_action,
            "root_cause": self.root_cause,
            "corrective_action": self.corrective_action,
            "preventive_action": self.preventive_action,
            "verification_method": self.verification_method,
            "verification_result": self.verification_result,
            "effectiveness_check_date": (str(self.effectiveness_check_date) if self.effectiveness_check_date else None),
            "recurrence_check": self.is_recurrence_checked,
            "cost_of_poor_quality": (str(self.cost_of_poor_quality) if self.cost_of_poor_quality else None),
            "copq_category": self.copq_category,
            "copq_paf_class": self.copq_paf_class,
            "due_date": str(self.due_date) if self.due_date else None,
            "assigned_to": None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "project_id": str(self.project_id) if self.project_id else None,
            "rca_session_id": str(self.rca_session_id) if self.rca_session_id else None,
        }
        if self.assigned_to:
            d["assigned_to"] = {
                "id": self.assigned_to_id,
                "name": getattr(self.assigned_to, "display_name", "") or self.assigned_to.email,
            }
        try:
            d["status_changes"] = [sc.to_dict() for sc in self.status_changes.order_by("created_at")]
        except Exception:
            d["status_changes"] = []
        return d


class CAPAStatusChange(models.Model):
    """Status change history for CAPA reports."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    capa = models.ForeignKey(
        CAPAReport,
        on_delete=models.CASCADE,
        related_name="status_changes",
    )
    from_status = models.CharField(max_length=20)
    to_status = models.CharField(max_length=20)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_capa_status_changes"
        ordering = ["created_at"]

    def to_dict(self):
        return {
            "id": str(self.id),
            "from_status": self.from_status,
            "to_status": self.to_status,
            "changed_by": ((self.changed_by.display_name or self.changed_by.email) if self.changed_by else None),
            "note": self.note,
            "created_at": self.created_at.isoformat(),
        }


class InternalAudit(models.Model):
    """Internal audit scheduler per ISO 9001 clause 9.2."""

    class Status(models.TextChoices):
        PLANNED = "planned", "Planned"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETE = "complete", "Complete"
        REPORT_ISSUED = "report_issued", "Report Issued"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="internal_audits",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_records",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="audits_created",
    )
    title = models.CharField(max_length=300)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PLANNED)
    scheduled_date = models.DateField()
    completed_date = models.DateField(null=True, blank=True)
    lead_auditor = models.CharField(max_length=200, blank=True)
    iso_clauses = models.JSONField(default=list, blank=True, help_text="List of clause refs")
    departments = models.JSONField(default=list, blank=True)
    scope = models.TextField(blank=True)
    summary = models.TextField(blank=True)
    checklist_results = models.JSONField(
        default=dict,
        blank=True,
        help_text='{"<checklist_id>": {"checklist_name": str, "items": [{"question", "result", "notes"}], "completed_at": str}}',
    )
    # Graph linkage (GRAPH-001) — process areas being audited
    linked_process_node_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="UUIDs of ProcessNode records in scope of this audit — opt-in",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_audits"
        ordering = ["scheduled_date"]

    def __str__(self):
        return f"Audit: {self.title} ({self.scheduled_date})"

    def to_dict(self):
        d = {
            "id": str(self.id),
            "title": self.title,
            "status": self.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "scheduled_date": str(self.scheduled_date),
            "completed_date": str(self.completed_date) if self.completed_date else None,
            "lead_auditor": self.lead_auditor,
            "iso_clauses": self.iso_clauses,
            "departments": self.departments,
            "scope": self.scope,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings.all()],
            "checklist_results": self.checklist_results,
            "created_at": self.created_at.isoformat(),
            "field_changes": [],
        }
        try:
            d["field_changes"] = [
                fc.to_dict()
                for fc in QMSFieldChange.objects.filter(record_type="audit", record_id=self.id).select_related(
                    "changed_by"
                )[:50]
            ]
        except Exception:
            pass
        return d


class AuditFinding(models.Model):
    """Finding from an internal audit."""

    class FindingType(models.TextChoices):
        NC_MAJOR = "nc_major", "Major Nonconformity"
        NC_MINOR = "nc_minor", "Minor Nonconformity"
        OBSERVATION = "observation", "Observation"
        OPPORTUNITY = "opportunity", "Opportunity for Improvement"

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        CORRECTIVE_ACTION_REQUIRED = (
            "corrective_action_required",
            "Corrective Action Required",
        )
        CLOSED = "closed", "Closed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audit = models.ForeignKey(InternalAudit, on_delete=models.CASCADE, related_name="findings")
    finding_type = models.CharField(max_length=20, choices=FindingType.choices)
    description = models.TextField()
    iso_clause = models.CharField(max_length=20, blank=True)
    evidence = models.TextField(blank=True)
    corrective_action = models.TextField(blank=True)
    due_date = models.DateField(null=True, blank=True)
    is_resolved = models.BooleanField(default=False, db_column="resolved")
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.OPEN)
    created_at = models.DateTimeField(auto_now_add=True)

    # Optional link to NCR created from this finding
    ncr = models.ForeignKey(
        NonconformanceRecord,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_findings",
    )

    class Meta:
        db_table = "iso_audit_findings"
        ordering = ["-created_at"]

    def to_dict(self):
        return {
            "id": str(self.id),
            "finding_type": self.finding_type,
            "description": self.description,
            "iso_clause": self.iso_clause,
            "evidence": self.evidence,
            "corrective_action": self.corrective_action,
            "due_date": str(self.due_date) if self.due_date else None,
            "resolved": self.is_resolved,
            "status": self.status,
            "ncr_id": str(self.ncr_id) if self.ncr_id else None,
        }


class TrainingRequirement(models.Model):
    """Training requirement per ISO 9001 clause 7.2."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="training_requirements",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="training_records",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="training_created",
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    iso_clause = models.CharField(max_length=20, blank=True, help_text="e.g. 7.2, 8.5.1")
    frequency_months = models.IntegerField(default=0, help_text="0 = one-time")
    is_mandatory = models.BooleanField(default=False)
    document = models.ForeignKey(
        "agents_api.ControlledDocument",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="training_requirements",
        help_text="Linked controlled document (nullable for informal training)",
    )
    document_version = models.CharField(
        max_length=20,
        blank=True,
        help_text="Document version when training was last aligned",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_training_requirements"
        ordering = ["name"]

    def __str__(self):
        return self.name

    def to_dict(self):
        records = list(self.records.all())
        total = len(records)
        complete = sum(1 for r in records if r.status == TrainingRecord.Status.COMPLETE)
        expiring = sum(1 for r in records if r.is_expiring_soon())
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "iso_clause": self.iso_clause,
            "frequency_months": self.frequency_months,
            "is_mandatory": self.is_mandatory,
            "document_id": str(self.document_id) if self.document_id else None,
            "document_title": str(self.document) if self.document_id else None,
            "document_version": self.document_version,
            "completion_rate": round(complete / total * 100) if total else 0,
            "expiring_soon": expiring,
            "records": [r.to_dict() for r in records],
        }


class TrainingRecord(models.Model):
    """Individual training record for an employee."""

    class Status(models.TextChoices):
        NOT_STARTED = "not_started", "Not Started"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETE = "complete", "Complete"
        EXPIRED = "expired", "Expired"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    requirement = models.ForeignKey(TrainingRequirement, on_delete=models.CASCADE, related_name="records")
    employee_name = models.CharField(max_length=200)
    employee_email = models.EmailField(blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.NOT_STARTED)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)
    competency_level = models.IntegerField(
        default=0,
        help_text="TWI competency: 0=None, 1=Awareness, 2=Supervised, 3=Competent, 4=Trainer",
    )
    artifacts = models.ManyToManyField(
        "files.UserFile",
        blank=True,
        related_name="training_records",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_training_records"
        ordering = ["employee_name"]

    def is_expiring_soon(self):
        if not self.expires_at:
            return False
        from datetime import timedelta

        from django.utils import timezone

        return self.expires_at <= timezone.now() + timedelta(days=30)

    @property
    def certification_status(self):
        """Compute certification status from status + expires_at (TRN-001 §3)."""
        from datetime import timedelta

        from django.utils import timezone

        if self.status in ("not_started", "in_progress"):
            return "incomplete"
        if self.status == "expired":
            return "expired"
        if self.status == "complete":
            if self.expires_at and self.expires_at <= timezone.now():
                return "expired"
            if self.expires_at and self.expires_at <= timezone.now() + timedelta(days=30):
                return "expiring"
            return "current"
        return "incomplete"

    def to_dict(self):
        d = {
            "id": str(self.id),
            "employee_name": self.employee_name,
            "employee_email": self.employee_email,
            "status": self.status,
            "competency_level": self.competency_level,
            "certification_status": self.certification_status,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "notes": self.notes,
            "artifact_ids": ([str(f.id) for f in self.artifacts.all()] if self.pk else []),
            "changes": [],
        }
        try:
            d["changes"] = [c.to_dict() for c in self.changes.order_by("-created_at")]
        except Exception:
            pass
        return d


class TrainingRecordChange(models.Model):
    """Field-level change log for training records."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    record = models.ForeignKey(
        TrainingRecord,
        on_delete=models.CASCADE,
        related_name="changes",
    )
    field_name = models.CharField(max_length=50)
    old_value = models.TextField(blank=True)
    new_value = models.TextField(blank=True)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_training_record_changes"
        ordering = ["created_at"]

    def _safe_changed_by(self):
        if not self.changed_by_id:
            return None
        try:
            u = self.changed_by
            return u.display_name or u.email
        except Exception:
            return str(self.changed_by_id)

    def to_dict(self):
        return {
            "id": str(self.id),
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changed_by": self._safe_changed_by(),
            "created_at": self.created_at.isoformat(),
        }


class ManagementReviewTemplate(models.Model):
    """Customizable management review template per ISO 9001 §9.3.

    Defines which sections appear in a review, their order, and whether
    they auto-populate from QMS data or require manual input.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="review_templates",
    )
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    sections = models.JSONField(
        default=list,
        help_text='[{"key": str, "title": str, "data_source": "auto"|"manual", "auto_query": str|null, "required": bool}]',
    )
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_review_templates"
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

    def to_dict(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "sections": self.sections,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
        }

    # ISO 9001 §9.3.2 default sections
    DEFAULT_SECTIONS = [
        {
            "key": "prior_actions",
            "title": "Status of Prior Actions",
            "data_source": "auto",
            "auto_query": "prior_actions",
            "required": True,
        },
        {
            "key": "external_internal_issues",
            "title": "Changes in External/Internal Issues",
            "data_source": "manual",
            "auto_query": None,
            "required": True,
        },
        {
            "key": "customer_feedback",
            "title": "Customer Satisfaction & Feedback",
            "data_source": "manual",
            "auto_query": None,
            "required": True,
        },
        {
            "key": "process_performance",
            "title": "Process Performance & Conformity",
            "data_source": "manual",
            "auto_query": None,
            "required": True,
        },
        {
            "key": "ncr_corrective",
            "title": "Nonconformities & Corrective Actions",
            "data_source": "auto",
            "auto_query": "ncr_summary",
            "required": True,
        },
        {
            "key": "monitoring_results",
            "title": "Monitoring & Measurement Results",
            "data_source": "manual",
            "auto_query": None,
            "required": False,
        },
        {
            "key": "audit_results",
            "title": "Audit Results",
            "data_source": "auto",
            "auto_query": "audit_summary",
            "required": True,
        },
        {
            "key": "supplier_performance",
            "title": "Supplier Performance",
            "data_source": "manual",
            "auto_query": None,
            "required": False,
        },
        {
            "key": "resource_adequacy",
            "title": "Resource Adequacy",
            "data_source": "manual",
            "auto_query": None,
            "required": False,
        },
        {
            "key": "risk_opportunities",
            "title": "Risk & Opportunities Effectiveness",
            "data_source": "manual",
            "auto_query": None,
            "required": True,
        },
    ]


class ManagementReview(models.Model):
    """Management review per ISO 9001 clause 9.3."""

    class Status(models.TextChoices):
        SCHEDULED = "scheduled", "Scheduled"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETE = "complete", "Complete"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="management_reviews",
    )
    template = models.ForeignKey(
        ManagementReviewTemplate,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reviews",
    )
    title = models.CharField(max_length=300)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SCHEDULED)
    meeting_date = models.DateField()
    attendees = models.JSONField(default=list, blank=True, help_text="List of attendee names")
    # ISO 9001:2015 clause 9.3.2 inputs
    inputs = models.JSONField(
        default=dict,
        blank=True,
        help_text="Prior actions, audit results, customer feedback, process performance, NCRs, risks, opportunities",
    )
    # ISO 9001:2015 clause 9.3.3 outputs
    outputs = models.JSONField(
        default=dict,
        blank=True,
        help_text="Improvement opportunities, resource needs, changes to QMS",
    )
    minutes = models.TextField(blank=True)
    data_snapshot = models.JSONField(default=dict, blank=True, help_text="Auto-captured QMS metrics at review time")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_management_reviews"
        ordering = ["-meeting_date"]

    def __str__(self):
        return f"Review: {self.title} ({self.meeting_date})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "status": self.status,
            "meeting_date": str(self.meeting_date),
            "attendees": self.attendees,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "minutes": self.minutes,
            "data_snapshot": self.data_snapshot,
            "template_id": str(self.template_id) if self.template_id else None,
            "template_title": self.template.title if self.template else None,
            "created_at": self.created_at.isoformat(),
        }


class ControlledDocument(models.Model):
    """Document control per ISO 9001 clause 7.5."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        REVIEW = "review", "Under Review"
        APPROVED = "approved", "Approved"
        OBSOLETE = "obsolete", "Obsolete"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="controlled_documents",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="document_records",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="documents_created",
    )
    title = models.CharField(max_length=300)
    document_number = models.CharField(max_length=50, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    category = models.CharField(max_length=100, blank=True, help_text="e.g. SOP, Work Instruction, Policy")
    iso_clause = models.CharField(max_length=20, blank=True)
    current_version = models.CharField(max_length=20, default="1.0")
    review_due_date = models.DateField(null=True, blank=True)
    approved_by = models.CharField(max_length=200, blank=True)
    approved_by_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_documents",
    )
    approved_at = models.DateTimeField(null=True, blank=True)
    content = models.TextField(blank=True)
    retention_years = models.IntegerField(default=7, help_text="Document retention period in years")
    metadata = models.JSONField(default=dict, blank=True)
    # Graph linkage (GRAPH-001 §15.3) — process nodes this document covers
    linked_process_nodes = models.ManyToManyField(
        "graph.ProcessNode",
        blank=True,
        related_name="controlled_documents",
        help_text="ProcessNodes this document's procedures cover — enables PC→graph evidence",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Evidence attachments
    files = models.ManyToManyField(
        "files.UserFile",
        blank=True,
        related_name="controlled_documents",
    )

    # Study linkage — set when created via "Request Document Update" action
    source_study = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="controlled_documents",
    )

    # Valid status transitions
    TRANSITIONS = {
        "draft": {"review"},
        "review": {"approved", "draft"},
        "approved": {"review", "obsolete"},
        "obsolete": set(),
    }
    TRANSITION_REQUIRES = {
        "review": ["content"],
        "approved": ["approved_by_user"],
    }

    class Meta:
        db_table = "iso_controlled_documents"
        ordering = ["document_number", "title"]

    def __str__(self):
        return f"{self.document_number} - {self.title}" if self.document_number else self.title

    def can_transition(self, new_status):
        """Check if transition is valid and requirements are met."""
        allowed = self.TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            return False, f"Cannot transition from '{self.status}' to '{new_status}'"
        for field in self.TRANSITION_REQUIRES.get(new_status, []):
            fk_val = getattr(self, f"{field}_id", None)
            val = getattr(self, field, None)
            if fk_val is None and not val:
                return False, f"'{field}' is required to transition to '{new_status}'"
        return True, ""

    def to_dict(self):
        d = {
            "id": str(self.id),
            "title": self.title,
            "document_number": self.document_number,
            "status": self.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "category": self.category,
            "iso_clause": self.iso_clause,
            "current_version": self.current_version,
            "review_due_date": (str(self.review_due_date) if self.review_due_date else None),
            "approved_by": self.approved_by,
            "approved_by_user": None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "content": self.content,
            "retention_years": self.retention_years,
            "source_study_id": (str(self.source_study_id) if self.source_study_id else None),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_ids": [],
            "status_changes": [],
            "revisions": [],
        }
        if self.approved_by_user_id:
            try:
                u = self.approved_by_user
                name = u.display_name or u.email
            except Exception:
                name = str(self.approved_by_user_id)
            d["approved_by_user"] = {"id": self.approved_by_user_id, "name": name}
        try:
            d["file_ids"] = [str(f.id) for f in self.files.all()]
        except Exception:
            pass
        try:
            d["status_changes"] = [sc.to_dict() for sc in self.status_changes.order_by("created_at")]
        except Exception:
            pass
        try:
            d["revisions"] = [r.to_dict() for r in self.revisions.order_by("-created_at")[:20]]
        except Exception:
            pass
        try:
            d["field_changes"] = [
                fc.to_dict()
                for fc in QMSFieldChange.objects.filter(record_type="document", record_id=self.id).select_related(
                    "changed_by"
                )[:50]
            ]
        except Exception:
            d["field_changes"] = []
        return d


class DocumentRevision(models.Model):
    """Version history for controlled documents."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        ControlledDocument,
        on_delete=models.CASCADE,
        related_name="revisions",
    )
    version = models.CharField(max_length=20)
    change_summary = models.TextField(blank=True)
    content_snapshot = models.TextField(blank=True, help_text="Content at this revision")
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_document_revisions"
        ordering = ["-created_at"]

    def _safe_changed_by(self):
        if not self.changed_by_id:
            return None
        try:
            u = self.changed_by
            return u.display_name or u.email
        except Exception:
            return str(self.changed_by_id)

    def to_dict(self):
        return {
            "id": str(self.id),
            "version": self.version,
            "change_summary": self.change_summary,
            "changed_by": self._safe_changed_by(),
            "created_at": self.created_at.isoformat(),
        }


class DocumentStatusChange(models.Model):
    """Status change history for controlled documents."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        ControlledDocument,
        on_delete=models.CASCADE,
        related_name="status_changes",
    )
    from_status = models.CharField(max_length=20)
    to_status = models.CharField(max_length=20)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_document_status_changes"
        ordering = ["created_at"]

    def _safe_changed_by(self):
        if not self.changed_by_id:
            return None
        try:
            u = self.changed_by
            return u.display_name or u.email
        except Exception:
            return str(self.changed_by_id)

    def to_dict(self):
        return {
            "id": str(self.id),
            "from_status": self.from_status,
            "to_status": self.to_status,
            "changed_by": self._safe_changed_by(),
            "note": self.note,
            "created_at": self.created_at.isoformat(),
        }


class SupplierRecord(models.Model):
    """Supplier management per ISO 9001 clause 8.4."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending Approval"
        APPROVED = "approved", "Approved"
        PREFERRED = "preferred", "Preferred"
        CONDITIONAL = "conditional", "Conditional"
        SUSPENDED = "suspended", "Suspended"
        DISQUALIFIED = "disqualified", "Disqualified"

    SUPPLIER_TYPE_CHOICES = [
        ("raw_material", "Raw Material"),
        ("component", "Component"),
        ("service", "Service"),
        ("equipment", "Equipment"),
        ("calibration", "Calibration"),
        ("other", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="supplier_records",
    )
    name = models.CharField(max_length=300)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    supplier_type = models.CharField(max_length=30, choices=SUPPLIER_TYPE_CHOICES, default="other")
    contact_name = models.CharField(max_length=200, blank=True)
    contact_email = models.EmailField(blank=True)
    contact_phone = models.CharField(max_length=50, blank=True)
    products_services = models.TextField(blank=True, help_text="What they supply")
    iso_clause = models.CharField(max_length=20, blank=True)
    last_evaluation_date = models.DateField(null=True, blank=True)
    next_evaluation_date = models.DateField(null=True, blank=True)
    quality_rating = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
    )
    evaluation_scores = models.JSONField(
        default=dict,
        blank=True,
        help_text='{"quality": 1-5, "delivery": 1-5, "price": 1-5, "communication": 1-5}',
    )
    disqualification_reason = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Valid status transitions
    TRANSITIONS = {
        "pending": {"approved", "conditional", "disqualified"},
        "approved": {"preferred", "suspended", "disqualified"},
        "preferred": {"suspended", "disqualified"},
        "conditional": {"approved", "disqualified"},
        "suspended": {"approved", "disqualified"},
        "disqualified": {"pending"},
    }
    TRANSITION_REQUIRES = {
        "pending": ["notes"],  # A4: re-qualification rationale required
        "approved": ["quality_rating"],
        "preferred": ["quality_rating"],
        "conditional": ["notes"],
        "suspended": ["notes"],
        "disqualified": ["disqualification_reason"],
    }

    class Meta:
        db_table = "iso_suppliers"
        ordering = ["name"]

    def __str__(self):
        return self.name

    def can_transition(self, new_status):
        """Check if transition is valid and requirements are met."""
        allowed = self.TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            return False, f"Cannot transition from '{self.status}' to '{new_status}'"
        for field in self.TRANSITION_REQUIRES.get(new_status, []):
            val = getattr(self, field, None)
            if not val:
                return False, f"'{field}' is required to transition to '{new_status}'"
        return True, ""

    def to_dict(self):
        d = {
            "id": str(self.id),
            "name": self.name,
            "status": self.status,
            "supplier_type": self.supplier_type,
            "contact_name": self.contact_name,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "products_services": self.products_services,
            "quality_rating": self.quality_rating,
            "evaluation_scores": self.evaluation_scores,
            "disqualification_reason": self.disqualification_reason,
            "last_evaluation_date": (str(self.last_evaluation_date) if self.last_evaluation_date else None),
            "next_evaluation_date": (str(self.next_evaluation_date) if self.next_evaluation_date else None),
            "notes": self.notes,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status_changes": [],
        }
        try:
            d["status_changes"] = [sc.to_dict() for sc in self.status_changes.order_by("created_at")]
        except Exception:
            pass
        try:
            d["field_changes"] = [
                fc.to_dict()
                for fc in QMSFieldChange.objects.filter(record_type="supplier", record_id=self.id).select_related(
                    "changed_by"
                )[:50]
            ]
        except Exception:
            d["field_changes"] = []
        return d


class SupplierStatusChange(models.Model):
    """Status change history for suppliers."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    supplier = models.ForeignKey(
        SupplierRecord,
        on_delete=models.CASCADE,
        related_name="status_changes",
    )
    from_status = models.CharField(max_length=20)
    to_status = models.CharField(max_length=20)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_supplier_status_changes"
        ordering = ["created_at"]

    def _safe_changed_by(self):
        if not self.changed_by_id:
            return None
        try:
            u = self.changed_by
            return u.display_name or u.email
        except Exception:
            return str(self.changed_by_id)

    def to_dict(self):
        return {
            "id": str(self.id),
            "from_status": self.from_status,
            "to_status": self.to_status,
            "changed_by": self._safe_changed_by(),
            "note": self.note,
            "created_at": self.created_at.isoformat(),
        }


class QMSFieldChange(models.Model):
    """Field-level change log for QMS records (NCR, Audit, Document, Supplier)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    record_type = models.CharField(max_length=20, db_index=True)  # ncr, audit, document, supplier
    record_id = models.UUIDField(db_index=True)
    field_name = models.CharField(max_length=50)
    old_value = models.TextField(blank=True)
    new_value = models.TextField(blank=True)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "iso_qms_field_changes"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["record_type", "record_id"]),
        ]

    def _safe_changed_by(self):
        if not self.changed_by_id:
            return None
        try:
            u = self.changed_by
            return u.display_name or u.email
        except Exception:
            return str(self.changed_by_id)

    def to_dict(self):
        return {
            "id": str(self.id),
            "type": "field_change",
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changed_by": self._safe_changed_by(),
            "created_at": self.created_at.isoformat(),
        }


# =========================================================================
# C1: Customer Complaint Register — ISO 9001 §9.1.2
# =========================================================================


class CustomerComplaint(models.Model):
    """Customer complaint tracking per ISO 9001 §9.1.2.

    Tracks complaints through investigation to resolution with customer
    satisfaction follow-up. Links to NCR and CAPA for closed-loop action.
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        INVESTIGATING = "investigating", "Investigating"
        RESOLVED = "resolved", "Resolved"
        CLOSED = "closed", "Closed"

    class Source(models.TextChoices):
        PHONE = "phone", "Phone"
        EMAIL = "email", "Email"
        WEB = "web", "Web Form"
        FIELD = "field", "Field Report"
        SOCIAL = "social", "Social Media"
        RETURN = "return", "Product Return"
        OTHER = "other", "Other"

    TRANSITIONS = {
        "open": {"acknowledged"},
        "acknowledged": {"open", "investigating"},
        "investigating": {"acknowledged", "resolved"},
        "resolved": {"investigating", "closed"},
        "closed": {"resolved"},
    }
    TRANSITION_REQUIRES = {
        "acknowledged": ["assigned_to"],
        "resolved": ["resolution"],
        "closed": ["satisfaction_followup"],
    }
    # Reopening from closed requires explicit reason
    REOPEN_REQUIRES = {"resolved": ["reopen_reason"]}

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="complaints",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="complaints",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="complaints_created",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    source = models.CharField(max_length=20, choices=Source.choices, default=Source.OTHER)
    severity = models.CharField(
        max_length=20,
        choices=[
            ("low", "Low"),
            ("medium", "Medium"),
            ("high", "High"),
            ("critical", "Critical"),
        ],
        default="medium",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.OPEN)
    product_service = models.CharField(max_length=300, blank=True, help_text="Product or service complained about")
    customer_name = models.CharField(max_length=300, blank=True)
    customer_contact = models.CharField(max_length=300, blank=True, help_text="Email or phone")
    date_received = models.DateField(null=True, blank=True)
    date_acknowledged = models.DateField(null=True, blank=True)
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_complaints",
    )
    root_cause = models.TextField(blank=True)
    resolution = models.TextField(blank=True)
    preventive_action = models.TextField(blank=True)
    satisfaction_followup = models.TextField(blank=True, help_text="Customer satisfaction check after resolution")
    reopen_reason = models.TextField(blank=True, help_text="Required when reopening a closed complaint")
    customer_satisfied = models.BooleanField(null=True, blank=True)
    # Cross-tool links
    ncr = models.ForeignKey(
        "agents_api.NonconformanceRecord",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="complaints",
    )
    capa = models.ForeignKey(
        "agents_api.CAPAReport",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="complaints",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="complaints",
    )
    iso_clause = models.CharField(max_length=20, blank=True, default="9.1.2")
    # Graph linkage (GRAPH-001) — which process graph nodes this complaint relates to
    linked_process_node_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="UUIDs of ProcessNode records this complaint relates to — opt-in",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_customer_complaints"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Complaint: {self.title}"

    def can_transition(self, new_status):
        """Check if transition is valid and requirements are met."""
        allowed = self.TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            return False, f"Cannot transition from '{self.status}' to '{new_status}'"
        for field in self.TRANSITION_REQUIRES.get(new_status, []):
            val = getattr(self, field, None)
            if val is None:
                return False, f"'{field}' is required to transition to '{new_status}'"
            if isinstance(val, str) and not val.strip():
                return False, f"'{field}' is required to transition to '{new_status}'"
        # Reopening from closed requires explicit reason
        if self.status == "closed":
            for field in self.REOPEN_REQUIRES.get(new_status, []):
                val = getattr(self, field, None)
                if not val or (isinstance(val, str) and not val.strip()):
                    return False, f"'{field}' is required to reopen a closed complaint"
        return True, ""

    def to_dict(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "severity": self.severity,
            "status": self.status,
            "product_service": self.product_service,
            "customer_name": self.customer_name,
            "customer_contact": self.customer_contact,
            "date_received": str(self.date_received) if self.date_received else None,
            "date_acknowledged": (str(self.date_acknowledged) if self.date_acknowledged else None),
            "assigned_to": str(self.assigned_to_id) if self.assigned_to_id else None,
            "root_cause": self.root_cause,
            "resolution": self.resolution,
            "preventive_action": self.preventive_action,
            "satisfaction_followup": self.satisfaction_followup,
            "reopen_reason": self.reopen_reason,
            "customer_satisfied": self.customer_satisfied,
            "ncr_id": str(self.ncr_id) if self.ncr_id else None,
            "capa_id": str(self.capa_id) if self.capa_id else None,
            "project_id": str(self.project_id) if self.project_id else None,
            "site_id": str(self.site_id) if self.site_id else None,
            "iso_clause": self.iso_clause,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =========================================================================
# C2: Risk Register — ISO 9001 §6.1
# =========================================================================


class Risk(models.Model):
    """Organizational risk and opportunity register per ISO 9001 §6.1.

    Tracks risks through identification, assessment, mitigation, and review.
    Separate from FMEA (product/process risk) — this covers organizational,
    compliance, strategic, and operational risks.
    """

    class Category(models.TextChoices):
        OPERATIONAL = "operational", "Operational"
        QUALITY = "quality", "Quality"
        COMPLIANCE = "compliance", "Compliance"
        STRATEGIC = "strategic", "Strategic"
        SAFETY = "safety", "Safety"
        FINANCIAL = "financial", "Financial"
        SUPPLY_CHAIN = "supply_chain", "Supply Chain"

    class Status(models.TextChoices):
        IDENTIFIED = "identified", "Identified"
        ASSESSING = "assessing", "Assessing"
        MITIGATING = "mitigating", "Mitigating"
        ACCEPTED = "accepted", "Accepted"
        CLOSED = "closed", "Closed"

    class RiskType(models.TextChoices):
        RISK = "risk", "Risk"
        OPPORTUNITY = "opportunity", "Opportunity"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="risks",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="risks",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="risks_created",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    risk_type = models.CharField(max_length=20, choices=RiskType.choices, default=RiskType.RISK)
    category = models.CharField(max_length=20, choices=Category.choices, default=Category.OPERATIONAL)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.IDENTIFIED)
    likelihood = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1=Rare, 5=Almost certain",
    )
    impact = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1=Negligible, 5=Catastrophic",
    )
    risk_score = models.IntegerField(default=1, help_text="likelihood × impact (computed)")
    risk_owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="owned_risks",
    )
    mitigation_actions = models.JSONField(
        default=list,
        blank=True,
        help_text='[{"action": str, "owner": str, "due_date": str, "status": str}]',
    )
    residual_likelihood = models.IntegerField(
        null=True, blank=True, validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    residual_impact = models.IntegerField(
        null=True, blank=True, validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    residual_risk_score = models.IntegerField(null=True, blank=True)
    review_date = models.DateField(null=True, blank=True, help_text="Next scheduled review")
    review_frequency_months = models.IntegerField(default=3, help_text="How often to review this risk")
    iso_clause = models.CharField(max_length=20, blank=True, default="6.1")
    source_type = models.CharField(
        max_length=30,
        blank=True,
        default="manual",
        help_text="How this risk was created: manual, fmea, audit, complaint, spc",
    )
    # Cross-tool links
    fmea = models.ForeignKey(
        "fmea.FMEA",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="risks",
    )
    fmea_row = models.ForeignKey(
        "fmea.FMEARow",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="risks",
        help_text="Source FMEA failure mode — scores normalized from S/O/D (1-10) to L/I (1-5)",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="risks",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_risks"
        ordering = ["-risk_score", "-created_at"]

    def __str__(self):
        return f"Risk: {self.title} (score={self.risk_score})"

    def save(self, *args, **kwargs):
        self.risk_score = self.likelihood * self.impact
        if self.residual_likelihood and self.residual_impact:
            self.residual_risk_score = self.residual_likelihood * self.residual_impact
        super().save(*args, **kwargs)

    def to_dict(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "risk_type": self.risk_type,
            "category": self.category,
            "status": self.status,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "risk_owner": str(self.risk_owner_id) if self.risk_owner_id else None,
            "mitigation_actions": self.mitigation_actions,
            "residual_likelihood": self.residual_likelihood,
            "residual_impact": self.residual_impact,
            "residual_risk_score": self.residual_risk_score,
            "review_date": str(self.review_date) if self.review_date else None,
            "review_frequency_months": self.review_frequency_months,
            "iso_clause": self.iso_clause,
            "source_type": self.source_type,
            "fmea_id": str(self.fmea_id) if self.fmea_id else None,
            "fmea_row_id": str(self.fmea_row_id) if self.fmea_row_id else None,
            "project_id": str(self.project_id) if self.project_id else None,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =========================================================================
# AFE — Authorization for Expenditure
# =========================================================================


class AFE(models.Model):
    """Authorization for Expenditure — capital project approval tracking.

    Flexible N-level approval chain (org-defined, not hardcoded).
    Links to HoshinProject for ROI tracking, Risk register for justification,
    FMEA for process risk context. Uses ElectronicSignature for CFR-compliant sign-off.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        SUBMITTED = "submitted", "Submitted"
        IN_REVIEW = "in_review", "In Review"
        APPROVED = "approved", "Approved"
        DENIED = "denied", "Denied"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="afes_created",
    )

    # Identification
    afe_number = models.CharField(max_length=50, blank=True, help_text="e.g. AFE-2026-001 (auto or manual)")
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    department = models.CharField(max_length=100, blank=True)

    # Financial
    estimated_cost = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    budgeted_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    actual_cost = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    expected_savings = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    payback_months = models.DecimalField(max_digits=6, decimal_places=1, null=True, blank=True)
    roi_percent = models.DecimalField(max_digits=7, decimal_places=2, null=True, blank=True)

    # Justification
    business_justification = models.TextField(blank=True)
    alternatives_considered = models.TextField(blank=True)
    quote_reference = models.CharField(max_length=200, blank=True)
    po_number = models.CharField(max_length=100, blank=True)
    denial_reason = models.TextField(blank=True)

    # Key dates
    submitted_date = models.DateField(null=True, blank=True)
    decision_date = models.DateField(null=True, blank=True)
    expected_completion = models.DateField(null=True, blank=True)
    actual_completion = models.DateField(null=True, blank=True)

    # Cross-tool links
    hoshin_project = models.ForeignKey(
        "hoshin.HoshinProject",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
    )
    risk = models.ForeignKey(
        "agents_api.Risk",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
        help_text="Risk register entry that justifies this expenditure",
    )
    fmea = models.ForeignKey(
        "fmea.FMEA",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
    )
    checklist = models.ForeignKey(
        "agents_api.Checklist",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
        help_text="Pre-approval checklist (safety review, environmental, etc.)",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afes",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "afes"
        ordering = ["-created_at"]

    def __str__(self):
        return f"AFE: {self.afe_number or self.title} (${self.estimated_cost})"

    @property
    def approval_progress(self):
        """Percentage of required approval levels that are approved."""
        levels = self.approval_levels.all()
        if not levels.exists():
            return 0
        required = levels.filter(is_required=True)
        if not required.exists():
            return 100
        approved = required.filter(status="approved").count()
        return int(approved / required.count() * 100)

    @property
    def current_level(self):
        """Next approval level that needs action."""
        return self.approval_levels.filter(status="pending").order_by("level_order").first()

    def to_dict(self):
        levels = list(self.approval_levels.order_by("level_order"))
        return {
            "id": str(self.id),
            "afe_number": self.afe_number,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "department": self.department,
            "estimated_cost": float(self.estimated_cost),
            "budgeted_amount": float(self.budgeted_amount),
            "actual_cost": float(self.actual_cost),
            "expected_savings": float(self.expected_savings),
            "payback_months": (float(self.payback_months) if self.payback_months else None),
            "roi_percent": float(self.roi_percent) if self.roi_percent else None,
            "business_justification": self.business_justification,
            "alternatives_considered": self.alternatives_considered,
            "quote_reference": self.quote_reference,
            "po_number": self.po_number,
            "denial_reason": self.denial_reason,
            "submitted_date": str(self.submitted_date) if self.submitted_date else None,
            "decision_date": str(self.decision_date) if self.decision_date else None,
            "expected_completion": (str(self.expected_completion) if self.expected_completion else None),
            "actual_completion": (str(self.actual_completion) if self.actual_completion else None),
            "hoshin_project_id": (str(self.hoshin_project_id) if self.hoshin_project_id else None),
            "risk_id": str(self.risk_id) if self.risk_id else None,
            "fmea_id": str(self.fmea_id) if self.fmea_id else None,
            "checklist_id": str(self.checklist_id) if self.checklist_id else None,
            "project_id": str(self.project_id) if self.project_id else None,
            "site_id": str(self.site_id) if self.site_id else None,
            "approval_progress": self.approval_progress,
            "approval_levels": [lv.to_dict() for lv in levels],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AFEApprovalLevel(models.Model):
    """Single approval step in an AFE's approval chain.

    N levels per AFE, org-defined. Each level can be required or optional,
    and has a cost threshold that auto-determines if it applies.
    Sign-off uses ElectronicSignature for CFR compliance.
    """

    class LevelStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        DENIED = "denied", "Denied"
        SKIPPED = "skipped", "Skipped (below threshold)"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    afe = models.ForeignKey(AFE, on_delete=models.CASCADE, related_name="approval_levels")

    # Level definition
    level_order = models.IntegerField(default=0, help_text="0=first approver, 1=second, etc.")
    level_name = models.CharField(max_length=100, help_text="e.g. Site Manager, BU Director, VP Finance")
    cost_threshold = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        help_text="This level required when estimated_cost >= threshold. 0 = always required.",
    )
    is_required = models.BooleanField(default=True)

    # Approver
    approver = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afe_approvals",
    )
    approver_name = models.CharField(max_length=200, blank=True, help_text="Display name (for non-users)")

    # Decision
    status = models.CharField(max_length=20, choices=LevelStatus.choices, default=LevelStatus.PENDING)
    comments = models.TextField(blank=True)
    decided_at = models.DateTimeField(null=True, blank=True)

    # ElectronicSignature link (CFR compliance)
    signature = models.ForeignKey(
        "agents_api.ElectronicSignature",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="afe_approval_levels",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "afe_approval_levels"
        ordering = ["level_order"]
        constraints = [
            models.UniqueConstraint(fields=["afe", "level_order"], name="unique_afe_level_order"),
        ]

    def __str__(self):
        return f"{self.level_name} ({self.status})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "level_order": self.level_order,
            "level_name": self.level_name,
            "cost_threshold": float(self.cost_threshold),
            "is_required": self.is_required,
            "approver_id": str(self.approver_id) if self.approver_id else None,
            "approver_name": self.approver_name,
            "status": self.status,
            "comments": self.comments,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "signature_id": str(self.signature_id) if self.signature_id else None,
        }


# =========================================================================
# C3: Calibration Equipment Register — ISO 9001 §7.1.5
# =========================================================================


class MeasurementEquipment(models.Model):
    """Measurement equipment register per ISO 9001 §7.1.5.2.

    Tracks calibration status, schedules, and links to Gage R&R studies.
    """

    class Status(models.TextChoices):
        IN_SERVICE = "in_service", "In Service"
        DUE = "due", "Calibration Due"
        OVERDUE = "overdue", "Overdue"
        OUT_OF_CAL = "out_of_calibration", "Out of Calibration"
        OUT_OF_SERVICE = "out_of_service", "Out of Service"
        RETIRED = "retired", "Retired"

    class EquipmentType(models.TextChoices):
        DIMENSIONAL = "dimensional", "Dimensional"
        FORCE_TORQUE = "force_torque", "Force/Torque"
        TEMPERATURE = "temperature", "Temperature"
        PRESSURE = "pressure", "Pressure"
        ELECTRICAL = "electrical", "Electrical"
        MASS = "mass", "Mass/Weight"
        OPTICAL = "optical", "Optical"
        CHEMICAL = "chemical", "Chemical"
        OTHER = "other", "Other"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="equipment",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="equipment",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="equipment_created",
    )
    name = models.CharField(max_length=300)
    asset_id = models.CharField(max_length=100, blank=True, help_text="Internal asset tag")
    serial_number = models.CharField(max_length=100, blank=True)
    manufacturer = models.CharField(max_length=200, blank=True)
    model_number = models.CharField(max_length=200, blank=True)
    equipment_type = models.CharField(max_length=20, choices=EquipmentType.choices, default=EquipmentType.OTHER)
    location = models.CharField(max_length=300, blank=True)
    status = models.CharField(max_length=25, choices=Status.choices, default=Status.IN_SERVICE)
    # Calibration tracking
    calibration_interval_months = models.IntegerField(default=12)
    last_calibration_date = models.DateField(null=True, blank=True)
    next_calibration_due = models.DateField(null=True, blank=True)
    calibration_provider = models.CharField(max_length=300, blank=True)
    calibration_certificate = models.CharField(max_length=300, blank=True, help_text="Certificate number or reference")
    measurement_range = models.CharField(max_length=200, blank=True, help_text="e.g. 0-25mm, 0-100°C")
    resolution = models.CharField(max_length=100, blank=True, help_text="e.g. 0.001mm, 0.1°C")
    accuracy = models.CharField(max_length=100, blank=True, help_text="e.g. ±0.002mm")
    # Gage R&R link
    gage_studies = models.JSONField(
        default=list,
        blank=True,
        help_text="List of DSWResult IDs from Gage R&R studies",
    )
    # Graph linkage (GRAPH-001 §3.2) — this equipment IS a measurement node
    linked_process_node = models.ForeignKey(
        "graph.ProcessNode",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="measurement_equipment",
        help_text="ProcessNode of type 'measurement' or 'equipment' this record represents",
    )
    # Reliability (OLR-001 §11.2)
    mtbf_hours = models.FloatField(null=True, blank=True, help_text="Mean time between failures (hours)")
    weibull_shape = models.FloatField(null=True, blank=True, help_text="Weibull shape parameter (beta)")
    weibull_scale = models.FloatField(null=True, blank=True, help_text="Weibull scale parameter (eta)")
    failure_count = models.IntegerField(default=0, help_text="Historical failure count")
    failure_history = models.JSONField(default=list, blank=True, help_text="[{date, hours_at_failure, description}]")
    # Measurement uncertainty from Gage R&R (OLR-001 §11.3)
    measurement_uncertainty_percent = models.FloatField(
        null=True, blank=True, help_text="% contribution to total variation from Gage R&R"
    )
    notes = models.TextField(blank=True)
    iso_clause = models.CharField(max_length=20, blank=True, default="7.1.5")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_measurement_equipment"
        ordering = ["next_calibration_due", "name"]

    def __str__(self):
        return f"{self.name} ({self.asset_id or self.serial_number or 'no ID'})"

    @property
    def is_overdue(self):
        from datetime import date

        if self.next_calibration_due:
            return self.next_calibration_due < date.today()
        return False

    @property
    def is_due_soon(self):
        from datetime import date, timedelta

        if self.next_calibration_due:
            return self.next_calibration_due <= date.today() + timedelta(days=30)
        return False

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "asset_id": self.asset_id,
            "serial_number": self.serial_number,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "equipment_type": self.equipment_type,
            "location": self.location,
            "status": self.status,
            "calibration_interval_months": self.calibration_interval_months,
            "last_calibration_date": (str(self.last_calibration_date) if self.last_calibration_date else None),
            "next_calibration_due": (str(self.next_calibration_due) if self.next_calibration_due else None),
            "calibration_provider": self.calibration_provider,
            "calibration_certificate": self.calibration_certificate,
            "measurement_range": self.measurement_range,
            "resolution": self.resolution,
            "accuracy": self.accuracy,
            "gage_studies": self.gage_studies,
            "notes": self.notes,
            "is_overdue": self.is_overdue,
            "is_due_soon": self.is_due_soon,
            "site_id": str(self.site_id) if self.site_id else None,
            "iso_clause": self.iso_clause,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    # AuditChecklist model removed in CR-0.6a — superseded by Checklist (0 rows in prod)


# =============================================================================
# Universal Checklists — prompt-response model (Gawande-style)
# =============================================================================


class Checklist(models.Model):
    """Reusable checklist template with typed prompt-response items.

    Follows checklist science (Gawande): each item is a specific prompt with
    an expected response type. Supports READ-DO and DO-CONFIRM patterns.
    Attachable to any entity (audit, project, kaizen, equipment, training, etc.).

    Item schema:
    [
        {
            "prompt": "Verify torque spec meets drawing requirement",
            "response_type": "pass_fail_na",   # yes_no | pass_fail_na | text | numeric | select | file | signature
            "guidance": "Reference drawing 12345-A rev C",  # optional help text
            "options": ["Option A", "Option B"],             # for select type only
            "required": true,                                 # is response mandatory
            "unit": "Nm",                                     # for numeric type
            "accept_min": 10.0,                               # for numeric — auto-flag out of spec
            "accept_max": 15.0,
        }
    ]
    """

    class ChecklistType(models.TextChoices):
        READ_DO = "read_do", "Read-Do"
        DO_CONFIRM = "do_confirm", "Do-Confirm"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="checklists")
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="checklists",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    checklist_type = models.CharField(max_length=20, choices=ChecklistType.choices, default=ChecklistType.READ_DO)
    category = models.CharField(
        max_length=50,
        blank=True,
        help_text="Grouping: audit, kaizen, safety, equipment, training, project, general",
    )
    version = models.CharField(max_length=20, default="1.0")
    items = models.JSONField(default=list, help_text="Array of typed prompt-response items")
    is_template = models.BooleanField(default=True, help_text="Templates are reusable; non-templates are one-offs")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "checklists"
        ordering = ["category", "name"]

    def __str__(self):
        return f"{self.name} ({self.category or 'general'})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "checklist_type": self.checklist_type,
            "category": self.category,
            "version": self.version,
            "items": self.items,
            "item_count": len(self.items) if self.items else 0,
            "is_template": self.is_template,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ChecklistExecution(models.Model):
    """Instance of executing a checklist against a specific entity.

    Links a Checklist template to any entity (audit, project, NCR, equipment, etc.)
    via entity_type + entity_id. Stores responses per item with evidence files.

    Response schema (matches items array by index):
    [
        {
            "value": "pass" | "fail" | "na" | "yes" | "no" | "text..." | 42.5 | null,
            "notes": "Operator comment",
            "file_ids": ["uuid", ...],    # evidence photos/documents
            "responded_by": "user@email",
            "responded_at": "ISO timestamp",
            "out_of_spec": false,          # auto-computed for numeric with accept_min/max
        }
    ]
    """

    class Status(models.TextChoices):
        NOT_STARTED = "not_started", "Not Started"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETE = "complete", "Complete"
        BLOCKED = "blocked", "Blocked"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    checklist = models.ForeignKey(Checklist, on_delete=models.CASCADE, related_name="executions")
    executor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="checklist_executions",
    )
    # Generic entity link — any module can use checklists
    entity_type = models.CharField(
        max_length=30,
        db_index=True,
        help_text="audit, project, kaizen, ncr, capa, equipment, training, supplier, document, general",
    )
    entity_id = models.UUIDField(db_index=True, help_text="UUID of the linked entity")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.NOT_STARTED)
    responses = models.JSONField(default=list, help_text="Array of responses matching checklist items by index")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "checklist_executions"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["entity_type", "entity_id"], name="idx_clexec_entity"),
        ]

    def __str__(self):
        return f"Execution of {self.checklist.name} on {self.entity_type}/{self.entity_id}"

    @property
    def progress(self):
        """Fraction of items with a non-null response."""
        if not self.responses:
            return 0
        answered = sum(1 for r in self.responses if r.get("value") is not None)
        total = len(self.checklist.items) if self.checklist.items else 1
        return round(answered / total * 100)

    @property
    def pass_count(self):
        return sum(1 for r in (self.responses or []) if r.get("value") in ("pass", "yes"))

    @property
    def fail_count(self):
        return sum(1 for r in (self.responses or []) if r.get("value") in ("fail", "no"))

    @property
    def out_of_spec_count(self):
        return sum(1 for r in (self.responses or []) if r.get("out_of_spec"))

    def to_dict(self):
        return {
            "id": str(self.id),
            "checklist_id": str(self.checklist_id),
            "checklist_name": self.checklist.name,
            "checklist_type": self.checklist.checklist_type,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id),
            "executor_id": str(self.executor_id) if self.executor_id else None,
            "status": self.status,
            "responses": self.responses,
            "progress": self.progress,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "out_of_spec_count": self.out_of_spec_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# ISO Document Creator (structured authoring)
# =============================================================================


class ISODocument(models.Model):
    """Structured ISO document authoring tool.

    Separate from ControlledDocument (which is the document control register).
    Users author structured documents here, then optionally publish to
    Document Control via the controlled_document link.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        IN_PROGRESS = "in_progress", "In Progress"
        REVIEW = "review", "Under Review"
        FINAL = "final", "Final"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="iso_documents",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="iso_documents",
    )
    document_type = models.CharField(
        max_length=30,
        help_text="Key into ISO_DOCUMENT_TYPES (e.g. 'procedure', 'work_instruction')",
    )
    title = models.CharField(max_length=300)
    document_number = models.CharField(max_length=50, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    version = models.CharField(max_length=20, default="1.0")
    iso_clause = models.CharField(max_length=20, blank=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Flexible metadata: effective_date, prepared_by, review_cycle, etc.",
    )
    controlled_document = models.OneToOneField(
        ControlledDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="source_iso_document",
    )
    files = models.ManyToManyField(
        "files.UserFile",
        blank=True,
        related_name="iso_documents",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_authored_documents"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.document_number} - {self.title}" if self.document_number else self.title

    def to_dict(self):
        d = {
            "id": str(self.id),
            "document_type": self.document_type,
            "title": self.title,
            "document_number": self.document_number,
            "status": self.status,
            "version": self.version,
            "iso_clause": self.iso_clause,
            "metadata": self.metadata,
            "controlled_document_id": (str(self.controlled_document_id) if self.controlled_document_id else None),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        try:
            d["section_count"] = self.sections.count()
        except Exception:
            d["section_count"] = 0
        try:
            d["file_ids"] = [str(f.id) for f in self.files.all()]
        except Exception:
            d["file_ids"] = []
        return d

    def to_dict_full(self):
        """Full dict including all sections, for detail/editor view."""
        d = self.to_dict()
        d["sections"] = [s.to_dict() for s in self.sections.order_by("sort_order")]
        return d


class ISOSection(models.Model):
    """Section within an ISODocument.

    section_type determines rendering and structured_data shape:
    - heading: level in structured_data
    - paragraph: content in content field
    - definition: term/definition pairs in structured_data
    - reference: cross-references in structured_data
    - image: file FK + caption
    - table: columns/rows in structured_data
    - checklist: items in structured_data
    - signature_block: signers in structured_data
    """

    class SectionType(models.TextChoices):
        HEADING = "heading", "Heading"
        PARAGRAPH = "paragraph", "Paragraph"
        DEFINITION = "definition", "Definition"
        REFERENCE = "reference", "Reference"
        IMAGE = "image", "Image"
        TABLE = "table", "Table"
        CHECKLIST = "checklist", "Checklist"
        SIGNATURE_BLOCK = "signature_block", "Signature Block"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        ISODocument,
        on_delete=models.CASCADE,
        related_name="sections",
    )
    sort_order = models.IntegerField(default=0)
    section_type = models.CharField(
        max_length=20,
        choices=SectionType.choices,
        default=SectionType.PARAGRAPH,
    )
    section_key = models.CharField(
        max_length=50,
        blank=True,
        help_text="Key from document type registry (blank for user-added sections)",
    )
    title = models.CharField(max_length=300, blank=True)
    content = models.TextField(blank=True, help_text="Text content for paragraph/heading sections")
    structured_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Type-specific data: table rows, checklist items, definition pairs, etc.",
    )
    image = models.ForeignKey(
        "files.UserFile",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="iso_section_images",
    )
    image_caption = models.CharField(max_length=500, blank=True)
    embedded_media = models.JSONField(
        default=list,
        blank=True,
        help_text='Whiteboard exports: [{"file_id": "...", "board_name": "...", "room_code": "...", "format": "svg"|"png"}]',
    )
    numbering = models.CharField(max_length=20, blank=True, help_text="e.g. 4.1, 4.1.2")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_authored_sections"
        ordering = ["sort_order"]

    def __str__(self):
        return f"{self.numbering} {self.title}" if self.numbering else self.title or f"Section {self.sort_order}"

    def to_dict(self):
        d = {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "sort_order": self.sort_order,
            "section_type": self.section_type,
            "section_key": self.section_key,
            "title": self.title,
            "content": self.content,
            "structured_data": self.structured_data,
            "image_id": str(self.image_id) if self.image_id else None,
            "image_caption": self.image_caption,
            "embedded_media": self.embedded_media,
            "numbering": self.numbering,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.image_id:
            try:
                d["image_url"] = self.image.file.url
            except Exception:
                d["image_url"] = None
        else:
            d["image_url"] = None
        return d


# =============================================================================
# Electronic Signatures — 21 CFR Part 11 (FEAT-003)
# =============================================================================


class ElectronicSignature(SynaraImmutableLog):
    """Electronic signature per FDA 21 CFR Part 11 and ISO 9001:2015 §7.5.3.

    Immutable record of a user's authenticated approval, rejection, review,
    authorship, or witnessing of a QMS document. Each signature:
    - Requires re-authentication (password re-entry) at signing time
    - Captures document state snapshot for traceability
    - Participates in hash chain for tamper detection
    - Records IP and user agent for forensic trail

    Compliance:
    - 21 CFR Part 11 §11.50: Signature manifestations (signer, date, meaning)
    - 21 CFR Part 11 §11.70: Signature/record linking
    - 21 CFR Part 11 §11.10(e): Audit trail integrity
    - 21 CFR Part 11 §11.10(c): Accurate record copies (document snapshot)
    - ISO 9001:2015 §7.5.3: Control of documented information
    """

    class Meaning(models.TextChoices):
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        REVIEWED = "reviewed", "Reviewed"
        AUTHORED = "authored", "Authored"
        WITNESSED = "witnessed", "Witnessed"

    # === Signature-specific fields ===

    signer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="electronic_signatures",
    )

    document_type = models.CharField(
        max_length=30,
        db_index=True,
        help_text="Signable type key (ncr, capa, document, review, audit, training, fmea)",
    )

    document_id = models.UUIDField(
        db_index=True,
        help_text="UUID of the signed document",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )

    meaning = models.CharField(
        max_length=20,
        choices=Meaning.choices,
        help_text="21 CFR Part 11 §11.50: signature meaning",
    )

    user_agent = models.TextField(
        blank=True,
        default="",
        help_text="Browser/client user agent at signing time",
    )

    class Meta(SynaraImmutableLog.Meta):
        db_table = "iso_electronic_signatures"
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["signer", "document_type", "document_id", "meaning"],
                name="unique_signature_per_meaning",
            ),
        ]
        indexes = [
            models.Index(
                fields=["document_type", "document_id"],
                name="idx_esig_doc",
            ),
            models.Index(
                fields=["signer", "created_at"],
                name="idx_esig_signer_time",
            ),
            models.Index(
                fields=["tenant_id", "created_at"],
                name="idx_esig_tenant_time",
            ),
        ]
        default_permissions = ("add", "view")

    def __str__(self):
        return f"E-Sig: {self.actor} {self.meaning} {self.document_type}/{self.document_id}"

    def _compute_hash_chain(self):
        """Override to add select_for_update for concurrent safety."""
        from django.db import transaction

        with transaction.atomic():
            previous = (
                self.__class__.objects.select_for_update()
                .filter(tenant_id=self.tenant_id)
                .order_by("-created_at")
                .first()
            )
            if previous:
                self.previous_hash = previous.entry_hash
            self.entry_hash = self._compute_entry_hash()

    def _compute_entry_hash(self) -> str:
        """Include signature-specific fields in hash for tamper detection."""
        import hashlib
        import json as _json

        data = {
            "correlation_id": str(self.correlation_id),
            "parent_correlation_id": (str(self.parent_correlation_id) if self.parent_correlation_id else ""),
            "tenant_id": str(self.tenant_id),
            "event_name": self.event_name,
            "actor": self.actor,
            "before_snapshot": _json.dumps(self.before_snapshot, sort_keys=True),
            "after_snapshot": _json.dumps(self.after_snapshot, sort_keys=True),
            "changes": _json.dumps(self.changes, sort_keys=True),
            "reason": self.reason,
            "previous_hash": self.previous_hash,
            # Signature-specific fields included in hash
            "signer_id": str(self.signer_id),
            "document_type": self.document_type,
            "document_id": str(self.document_id),
            "meaning": self.meaning,
        }
        data_json = _json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_json.encode("utf-8")).hexdigest()

    def to_dict(self):
        signer_name = ""
        if self.signer_id:
            try:
                u = self.signer
                signer_name = getattr(u, "display_name", "") or u.email
            except Exception:
                signer_name = str(self.signer_id)
        return {
            "id": str(self.id),
            "signer": {"id": self.signer_id, "name": signer_name},
            "document_type": self.document_type,
            "document_id": str(self.document_id),
            "meaning": self.meaning,
            "reason": self.reason,
            "actor_ip": self.actor_ip,
            "entry_hash": self.entry_hash,
            "previous_hash": self.previous_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =========================================================================
# QMS Attachments (ISO 9001 §7.5 — Documented Information)
# =========================================================================


class QMSAttachment(models.Model):
    """Generic attachment linking a UserFile to any QMS record.

    Uses entity_type + entity_id pattern to attach files to NCR, CAPA,
    FMEA, RCA, A3, ManagementReview, InternalAudit, or ControlledDocument.
    """

    class EntityType(models.TextChoices):
        NCR = "ncr", "Nonconformance Record"
        CAPA = "capa", "CAPA Report"
        FMEA = "fmea", "FMEA"  # noqa: F811
        RCA = "rca", "RCA Session"
        A3 = "a3", "A3 Report"
        MANAGEMENT_REVIEW = "management_review", "Management Review"
        INTERNAL_AUDIT = "audit", "Internal Audit"
        DOCUMENT = "document", "Controlled Document"

    class AttachmentType(models.TextChoices):
        EVIDENCE = "evidence", "Evidence"
        PHOTO = "photo", "Photo"
        REPORT = "report", "Report"
        FORM = "form", "Form"
        EXTERNAL = "external_document", "External Document"

    # Map entity_type to model class for validation
    ENTITY_MODEL_MAP = {
        "ncr": "NonconformanceRecord",
        "capa": "CAPAReport",
        "fmea": "FMEA",
        "rca": "RCASession",
        "a3": "A3Report",
        "management_review": "ManagementReview",
        "audit": "InternalAudit",
        "document": "ControlledDocument",
    }

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    entity_type = models.CharField(max_length=30, choices=EntityType.choices)
    entity_id = models.UUIDField()
    file = models.ForeignKey(
        "files.UserFile",
        on_delete=models.CASCADE,
        related_name="qms_attachments",
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="qms_attachments",
    )
    description = models.CharField(max_length=500, blank=True)
    attachment_type = models.CharField(
        max_length=30,
        choices=AttachmentType.choices,
        default=AttachmentType.EVIDENCE,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "qms_attachments"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["entity_type", "entity_id"]),
        ]

    def __str__(self):
        return f"{self.entity_type}:{self.entity_id} — {self.file.original_name}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id),
            "file": {
                "id": str(self.file_id),
                "name": self.file.original_name,
                "size_bytes": self.file.size_bytes,
                "mime_type": self.file.mime_type,
                "url": self.file.url,
            },
            "uploaded_by": self.uploaded_by_id,
            "description": self.description,
            "attachment_type": self.attachment_type,
            "created_at": self.created_at.isoformat(),
        }


# =========================================================================
# Control Plan — ISO 9001 §8.5.1 (Process Control)
# =========================================================================


class ControlPlan(models.Model):
    """Control Plan per ISO 9001 §8.5.1 / IATF 16949 §8.5.1.1.

    Defines what to monitor, how, how often, and what to do when out of spec.
    Each plan covers a process or process area. Items reference ProcessNodes
    from the graph when linked — but the plan works standalone too.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        SUPERSEDED = "superseded", "Superseded"

    class PlanType(models.TextChoices):
        PROTOTYPE = "prototype", "Prototype"
        PRE_LAUNCH = "pre_launch", "Pre-Launch"
        PRODUCTION = "production", "Production"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="control_plans",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="control_plans",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="%(app_label)s_%(class)s_set",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="control_plans_created",
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    plan_type = models.CharField(max_length=20, choices=PlanType.choices, default=PlanType.PRODUCTION)
    process_name = models.CharField(max_length=300, blank=True, help_text="Process or operation this plan covers")
    part_number = models.CharField(max_length=100, blank=True)
    revision = models.CharField(max_length=20, default="1.0")
    effective_date = models.DateField(null=True, blank=True)

    # Graph linkage — optional, plan works without it
    linked_graph = models.ForeignKey(
        "graph.ProcessGraph",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="control_plans",
        help_text="ProcessGraph this control plan monitors",
    )

    iso_clause = models.CharField(max_length=20, blank=True, default="8.5.1")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_control_plans"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.get_plan_type_display()}) — {self.status}"

    @property
    def item_count(self):
        return self.items.count()

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "plan_type": self.plan_type,
            "process_name": self.process_name,
            "part_number": self.part_number,
            "revision": self.revision,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "linked_graph_id": str(self.linked_graph_id) if self.linked_graph_id else None,
            "item_count": self.item_count,
            "iso_clause": self.iso_clause,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ControlPlanItem(models.Model):
    """A single characteristic monitored within a Control Plan.

    Each item defines: what to measure, how, how often, spec limits,
    and what to do when out of spec. Optionally links to a ProcessNode
    for graph integration.
    """

    class ControlMethod(models.TextChoices):
        SPC = "spc", "SPC Control Chart"
        INSPECTION = "inspection", "Inspection"
        GAGE = "gage", "Gage/Measurement"
        VISUAL = "visual", "Visual Check"
        FUNCTIONAL = "functional", "Functional Test"
        AUTOMATED = "automated", "Automated Monitoring"
        OTHER = "other", "Other"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    control_plan = models.ForeignKey(
        ControlPlan,
        on_delete=models.CASCADE,
        related_name="items",
    )

    # What to monitor
    process_step = models.CharField(max_length=300, help_text="Operation or process step name")
    characteristic = models.CharField(
        max_length=300, help_text="What is being measured (dimension, property, attribute)"
    )
    characteristic_type = models.CharField(
        max_length=20,
        choices=[("product", "Product"), ("process", "Process")],
        default="product",
    )
    is_special = models.BooleanField(
        default=False, help_text="Special characteristic (safety, regulatory, fit/function)"
    )

    # How to measure
    control_method = models.CharField(max_length=20, choices=ControlMethod.choices, default=ControlMethod.INSPECTION)
    measurement_technique = models.CharField(max_length=300, blank=True, help_text="Specific instrument or technique")
    sample_size = models.CharField(max_length=100, blank=True, help_text="e.g. '5 pieces', '100%', 'n=30'")
    frequency = models.CharField(max_length=100, blank=True, help_text="e.g. 'every hour', 'per lot', 'per shift'")

    # Spec limits
    spec_usl = models.FloatField(null=True, blank=True, help_text="Upper spec limit")
    spec_lsl = models.FloatField(null=True, blank=True, help_text="Lower spec limit")
    spec_target = models.FloatField(null=True, blank=True, help_text="Target value")
    spec_unit = models.CharField(max_length=50, blank=True)

    # Reaction plan
    reaction_plan = models.TextField(blank=True, help_text="What to do when out of spec: contain, sort, notify, adjust")

    # OLR-001 §9: detection mechanism level for this control item
    detection_mechanism_level = models.IntegerField(
        null=True,
        blank=True,
        help_text="1-8 per OLR-001 §9 detection hierarchy",
    )

    # OLR-001 §15: minimum competency stage to execute this control
    competency_stage_required = models.IntegerField(
        default=1,
        help_text="1=See, 2=Do, 3=Teach — minimum stage to execute this control item",
    )

    # FMIS linkage — traces control plan to knowledge structure
    fmis_row = models.ForeignKey(
        "loop.FMISRow",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="control_plan_items",
        help_text="FMIS row this control item monitors — links control plan to knowledge structure",
    )

    # Graph linkage — optional
    linked_process_node = models.ForeignKey(
        "graph.ProcessNode",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="control_plan_items",
        help_text="ProcessNode this item monitors",
    )
    linked_equipment = models.ForeignKey(
        "agents_api.MeasurementEquipment",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="control_plan_items",
        help_text="Measurement equipment used for this characteristic",
    )

    # Ordering within the plan
    sequence = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "iso_control_plan_items"
        ordering = ["control_plan", "sequence"]

    def __str__(self):
        special = " ★" if self.is_special else ""
        return f"{self.process_step}: {self.characteristic}{special}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "process_step": self.process_step,
            "characteristic": self.characteristic,
            "characteristic_type": self.characteristic_type,
            "is_special": self.is_special,
            "control_method": self.control_method,
            "measurement_technique": self.measurement_technique,
            "sample_size": self.sample_size,
            "frequency": self.frequency,
            "spec_usl": self.spec_usl,
            "spec_lsl": self.spec_lsl,
            "spec_target": self.spec_target,
            "spec_unit": self.spec_unit,
            "reaction_plan": self.reaction_plan,
            "linked_process_node_id": str(self.linked_process_node_id) if self.linked_process_node_id else None,
            "linked_equipment_id": str(self.linked_equipment_id) if self.linked_equipment_id else None,
            "sequence": self.sequence,
        }


def _current_year():
    """Default value for fiscal_year fields. Kept for old migration references."""
    from datetime import date

    return date.today().year

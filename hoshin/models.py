# hoshin models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed
#
# Site, Employee, SiteAccess: managed=False copies here.
# agents_api still owns these (chokepoint models referenced by QMS).

import secrets
import uuid
from datetime import date

from django.conf import settings
from django.db import models


def _current_year():
    return date.today().year


class Site(models.Model):
    """Manufacturing site within an enterprise tenant."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("core.Tenant", on_delete=models.CASCADE, related_name="hoshin_sites")
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=20, blank=True)
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
        managed = False
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
    """Per-site access control for Hoshin Kanri."""

    class SiteRole(models.TextChoices):
        VIEWER = "viewer", "Viewer"
        MEMBER = "member", "Member"
        ADMIN = "admin", "Admin"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.ForeignKey(Site, on_delete=models.CASCADE, related_name="hoshin_access_list")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="hoshin_site_access")
    role = models.CharField(max_length=20, choices=SiteRole.choices, default=SiteRole.MEMBER)
    granted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "hoshin_site_access"
        managed = False

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


class HoshinProject(models.Model):
    """CI improvement project extending core.Project with Hoshin Kanri tracking."""

    class ProjectClass(models.TextChoices):
        KAIZEN = "kaizen", "Kaizen Event"
        PROJECT = "project", "Extended Project"

    class ProjectType(models.TextChoices):
        MATERIAL = "material", "Material Savings"
        LABOR = "labor", "Labor Savings"
        QUALITY = "quality", "Quality Improvement"
        THROUGHPUT = "throughput", "Throughput Improvement"
        ENERGY = "energy", "Energy Reduction"
        SAFETY = "safety", "Safety Improvement"
        OTHER = "other", "Other"

    class Opportunity(models.TextChoices):
        CARRYOVER = "carryover", "Carryover from Prior Year"
        BUDGETED_NEW = "budgeted_new", "Budgeted New"
        CONTINGENCY = "contingency", "Contingency"
        UNPLANNED = "unplanned", "Unplanned/Reactive"

    class HoshinStatus(models.TextChoices):
        PROPOSED = "proposed", "Proposed"
        BUDGETED = "budgeted", "Budgeted"
        ACTIVE = "active", "Active"
        DELAYED = "delayed", "Delayed"
        COMPLETED = "completed", "Completed"
        ABORTED = "aborted", "Aborted"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.OneToOneField("core.Project", on_delete=models.CASCADE, related_name="hoshin_ext")
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_projects_ext")
    project_class = models.CharField(max_length=20, choices=ProjectClass.choices, default=ProjectClass.PROJECT)
    project_type = models.CharField(max_length=20, choices=ProjectType.choices, default=ProjectType.MATERIAL)
    opportunity = models.CharField(max_length=20, choices=Opportunity.choices, default=Opportunity.BUDGETED_NEW)
    hoshin_status = models.CharField(max_length=20, choices=HoshinStatus.choices, default=HoshinStatus.PROPOSED)
    fiscal_year = models.IntegerField(default=_current_year)
    annual_savings_target = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    calculation_method = models.CharField(max_length=30, blank=True)
    custom_formula = models.CharField(max_length=500, blank=True, default="")
    custom_formula_desc = models.CharField(max_length=200, blank=True, default="")
    kaizen_charter = models.JSONField(default=dict, blank=True)
    monthly_actuals = models.JSONField(default=list, blank=True)
    baseline_data = models.JSONField(default=list, blank=True)
    source_vsm = models.ForeignKey(
        "agents_api.ValueStreamMap",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="hoshin_projects_ext",
    )
    source_burst_id = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_projects"
        managed = False
        ordering = ["-updated_at"]

    def __str__(self):
        return f"[{self.get_project_class_display()}] {self.project.title}"

    @property
    def ytd_savings(self):
        return sum(m.get("savings", 0) or 0 for m in (self.monthly_actuals or []))

    @property
    def savings_pct(self):
        if not self.annual_savings_target:
            return 0
        return float(self.ytd_savings / float(self.annual_savings_target) * 100)

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "project_title": self.project.title,
            "project_status": self.project.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "site_name": self.site.name if self.site else None,
            "project_class": self.project_class,
            "project_type": self.project_type,
            "opportunity": self.opportunity,
            "hoshin_status": self.hoshin_status,
            "fiscal_year": self.fiscal_year,
            "annual_savings_target": float(self.annual_savings_target),
            "calculation_method": self.calculation_method,
            "custom_formula": self.custom_formula,
            "custom_formula_desc": self.custom_formula_desc,
            "kaizen_charter": self.kaizen_charter,
            "monthly_actuals": self.monthly_actuals,
            "baseline_data": self.baseline_data,
            "ytd_savings": self.ytd_savings,
            "savings_pct": self.savings_pct,
            "source_vsm_id": str(self.source_vsm_id) if self.source_vsm_id else None,
            "source_burst_id": self.source_burst_id,
            "champion_name": self.project.champion_name,
            "leader_name": self.project.leader_name,
            "team_members": self.project.team_members,
            "methodology": self.project.methodology,
            "current_phase": self.project.current_phase,
            "goal_metric": self.project.goal_metric,
            "goal_baseline": (str(self.project.goal_baseline) if self.project.goal_baseline else None),
            "goal_target": (str(self.project.goal_target) if self.project.goal_target else None),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ProjectTemplate(models.Model):
    """Reusable template for Hoshin/kaizen projects."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="hoshin_project_templates"
    )
    site = models.ForeignKey(
        "agents_api.Site", on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_project_templates"
    )
    tenant = models.ForeignKey(
        "core.Tenant", on_delete=models.CASCADE, null=True, blank=True, related_name="hoshin_project_templates"
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    project_class = models.CharField(max_length=20, choices=HoshinProject.ProjectClass.choices, default="project")
    project_type = models.CharField(max_length=20, choices=HoshinProject.ProjectType.choices, default="material")
    opportunity = models.CharField(max_length=20, choices=HoshinProject.Opportunity.choices, default="budgeted_new")
    calculation_method = models.CharField(max_length=30, blank=True)
    checklist_ids = models.JSONField(default=list, blank=True)
    default_actions = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "project_templates"
        managed = False
        ordering = ["name"]

    def __str__(self):
        return self.name

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "project_class": self.project_class,
            "project_type": self.project_type,
            "opportunity": self.opportunity,
            "calculation_method": self.calculation_method,
            "checklist_ids": self.checklist_ids,
            "checklist_count": len(self.checklist_ids) if self.checklist_ids else 0,
            "default_actions": self.default_actions,
            "action_count": len(self.default_actions) if self.default_actions else 0,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ActionItem(models.Model):
    """Task/action item with Gantt-style tracking."""

    class Status(models.TextChoices):
        NOT_STARTED = "not_started", "Not Started"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"
        BLOCKED = "blocked", "Blocked"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey("core.Project", on_delete=models.CASCADE, related_name="hoshin_action_items")
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    owner_name = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.NOT_STARTED)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    due_date = models.DateField(null=True, blank=True)
    progress = models.IntegerField(default=0)
    depends_on = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_dependents"
    )
    sort_order = models.IntegerField(default=0)
    source_type = models.CharField(max_length=20, blank=True, default="")
    source_id = models.UUIDField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "action_items"
        managed = False
        ordering = ["sort_order", "start_date"]

    def save(self, *args, **kwargs):
        if self.depends_on_id:
            if self.depends_on_id == self.pk:
                raise ValueError("ActionItem cannot depend on itself")
            visited = {self.pk}
            current_id = self.depends_on_id
            while current_id:
                if current_id in visited:
                    raise ValueError("Circular dependency detected in ActionItem chain")
                visited.add(current_id)
                try:
                    parent = ActionItem.objects.only("depends_on_id").get(pk=current_id)
                    current_id = parent.depends_on_id
                except ActionItem.DoesNotExist:
                    break
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "title": self.title,
            "description": self.description,
            "owner_name": self.owner_name,
            "status": self.status,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "progress": self.progress,
            "depends_on_id": str(self.depends_on_id) if self.depends_on_id else None,
            "sort_order": self.sort_order,
            "source_type": self.source_type,
            "source_id": str(self.source_id) if self.source_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Employee(models.Model):
    """CI participant / contact within a tenant."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("core.Tenant", on_delete=models.CASCADE, related_name="hoshin_employees")
    name = models.CharField(max_length=255)
    email = models.EmailField()
    role = models.CharField(max_length=255, blank=True)
    department = models.CharField(max_length=255, blank=True)
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_employees")
    user_link = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="hoshin_employee_profile",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_employees"
        managed = False

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


class ResourceCommitment(models.Model):
    """Employee assignment to a HoshinProject."""

    ROLE_CHOICES = [
        ("facilitator", "Facilitator"),
        ("team_member", "Team Member"),
        ("sponsor", "Sponsor"),
        ("process_owner", "Process Owner"),
        ("subject_expert", "Subject Expert"),
    ]
    STATUS_CHOICES = [
        ("requested", "Requested"),
        ("confirmed", "Confirmed"),
        ("active", "Active"),
        ("completed", "Completed"),
        ("declined", "Declined"),
    ]
    VALID_TRANSITIONS = {
        "requested": {"confirmed", "declined"},
        "confirmed": {"active", "declined"},
        "active": {"completed"},
    }

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name="hoshin_commitments")
    project = models.ForeignKey(HoshinProject, on_delete=models.CASCADE, related_name="hoshin_commitments")
    role = models.CharField(max_length=30, choices=ROLE_CHOICES)
    start_date = models.DateField()
    end_date = models.DateField()
    hours_per_day = models.DecimalField(max_digits=4, decimal_places=1, default=8)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="requested")
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="hoshin_requested_commitments"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_resource_commitments"
        managed = False

    def __str__(self):
        return f"{self.employee.name} → {self.project.project.title} ({self.role})"

    @classmethod
    def check_availability(cls, employee, start_date, end_date, exclude_id=None):
        qs = cls.objects.filter(
            employee=employee,
            start_date__lt=end_date,
            end_date__gt=start_date,
        ).exclude(status__in=("completed", "declined"))
        if exclude_id:
            qs = qs.exclude(pk=exclude_id)
        return qs

    def to_dict(self):
        return {
            "id": str(self.id),
            "employee_id": str(self.employee_id),
            "employee_name": self.employee.name,
            "project_id": str(self.project_id),
            "project_title": self.project.project.title,
            "role": self.role,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "hours_per_day": float(self.hours_per_day),
            "status": self.status,
            "requested_by_id": (str(self.requested_by_id) if self.requested_by_id else None),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ActionToken(models.Model):
    """Secure, time-limited, single-use, action-scoped access for non-users."""

    ACTION_CHOICES = [
        ("confirm_availability", "Confirm Availability"),
        ("decline", "Decline"),
        ("update_progress", "Update Progress"),
        ("view_dashboard", "View Dashboard"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name="hoshin_action_tokens")
    action_type = models.CharField(max_length=30, choices=ACTION_CHOICES)
    scoped_to = models.JSONField(default=dict)
    token = models.CharField(max_length=64, unique=True, db_index=True)
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "hoshin_action_tokens"
        managed = False

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        if not self.expires_at:
            from datetime import timedelta

            from django.utils import timezone

            self.expires_at = timezone.now() + timedelta(hours=72)
        super().save(*args, **kwargs)

    @property
    def is_valid(self):
        from django.utils import timezone

        return self.used_at is None and self.expires_at > timezone.now()

    def use(self):
        from django.utils import timezone

        self.used_at = timezone.now()
        self.save(update_fields=["used_at"])

    def __str__(self):
        return f"Token({self.action_type}) → {self.employee.name}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "employee_id": str(self.employee_id),
            "employee_name": self.employee.name,
            "action_type": self.action_type,
            "scoped_to": self.scoped_to,
            "is_valid": self.is_valid,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
        }


class StrategicObjective(models.Model):
    """3-5 year breakthrough goal. South quadrant of X-matrix."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        ACHIEVED = "achieved", "Achieved"
        DEFERRED = "deferred", "Deferred"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("core.Tenant", on_delete=models.CASCADE, related_name="hoshin_strategic_objectives")
    title = models.CharField(max_length=500)
    description = models.TextField(blank=True)
    owner_name = models.CharField(max_length=255, blank=True)
    start_year = models.IntegerField()
    end_year = models.IntegerField()
    target_metric = models.CharField(max_length=255, blank=True)
    target_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    target_unit = models.CharField(max_length=50, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    sort_order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_strategic_objectives"
        managed = False
        ordering = ["sort_order", "start_year"]

    def __str__(self):
        return f"{self.title} ({self.start_year}-{self.end_year})"

    def to_dict(self):
        from agents_api.models import HoshinKPI

        meta = HoshinKPI.METRIC_CATALOG.get(self.target_metric, {})
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "title": self.title,
            "description": self.description,
            "owner_name": self.owner_name,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "target_metric": self.target_metric,
            "metric_label": meta.get("label", self.target_metric or "—"),
            "metric_unit": meta.get("unit", self.target_unit or ""),
            "metric_aggregation": meta.get("aggregation", "sum"),
            "metric_direction": meta.get("direction", "up"),
            "target_value": float(self.target_value) if self.target_value else None,
            "target_unit": self.target_unit,
            "status": self.status,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AnnualObjective(models.Model):
    """This FY's specific target cascaded from a strategic objective. West quadrant."""

    class Status(models.TextChoices):
        ON_TRACK = "on_track", "On Track"
        AT_RISK = "at_risk", "At Risk"
        BEHIND = "behind", "Behind"
        ACHIEVED = "achieved", "Achieved"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("core.Tenant", on_delete=models.CASCADE, related_name="hoshin_annual_objectives")
    strategic_objective = models.ForeignKey(
        StrategicObjective, on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_annual_objectives"
    )
    site = models.ForeignKey(
        Site, on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_annual_objectives"
    )
    fiscal_year = models.IntegerField(default=_current_year)
    title = models.CharField(max_length=500)
    description = models.TextField(blank=True)
    owner_name = models.CharField(max_length=255, blank=True)
    target_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    actual_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    target_unit = models.CharField(max_length=50, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.ON_TRACK)
    sort_order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_annual_objectives"
        managed = False
        ordering = ["sort_order", "title"]

    def __str__(self):
        return f"FY{self.fiscal_year}: {self.title}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "strategic_objective_id": (str(self.strategic_objective_id) if self.strategic_objective_id else None),
            "site_id": str(self.site_id) if self.site_id else None,
            "site_name": self.site.name if self.site else None,
            "fiscal_year": self.fiscal_year,
            "title": self.title,
            "description": self.description,
            "owner_name": self.owner_name,
            "target_value": float(self.target_value) if self.target_value else None,
            "actual_value": float(self.actual_value) if self.actual_value else None,
            "target_unit": self.target_unit,
            "status": self.status,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class HoshinKPI(models.Model):
    """Measurable KPI for the East quadrant of the X-matrix."""

    class Frequency(models.TextChoices):
        MONTHLY = "monthly", "Monthly"
        QUARTERLY = "quarterly", "Quarterly"
        ANNUAL = "annual", "Annual"

    class Aggregation(models.TextChoices):
        SUM = "sum", "Sum (dollars)"
        WEIGHTED_AVG = "weighted_avg", "Volume-weighted average"
        LATEST = "latest", "Latest calculator result"
        MANUAL = "manual", "Manual entry"

    # Reference METRIC_CATALOG from agents_api.models.HoshinKPI
    # (keeping it there since it's a large constant shared with StrategicObjective.to_dict)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("core.Tenant", on_delete=models.CASCADE, related_name="hoshin_kpis_ext")
    fiscal_year = models.IntegerField(default=_current_year)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    target_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    actual_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)
    frequency = models.CharField(max_length=20, choices=Frequency.choices, default=Frequency.MONTHLY)
    direction = models.CharField(max_length=10, default="up")
    aggregation = models.CharField(max_length=20, choices=Aggregation.choices, default=Aggregation.SUM)
    derived_from = models.ForeignKey(
        HoshinProject, on_delete=models.SET_NULL, null=True, blank=True, related_name="hoshin_derived_kpis"
    )
    derived_field = models.CharField(max_length=30, blank=True, default="ytd_savings")
    calculator_result_type = models.CharField(max_length=60, blank=True, default="")
    calculator_field = models.CharField(max_length=60, blank=True, default="")
    sort_order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_kpis"
        managed = False
        ordering = ["sort_order", "name"]

    @property
    def effective_actual(self):
        """Return the KPI's actual value based on its aggregation mode."""
        if self.aggregation == "manual" or (not self.derived_from_id):
            return float(self.actual_value) if self.actual_value is not None else None
        proj = self.derived_from
        if not proj:
            return float(self.actual_value) if self.actual_value is not None else None
        if self.aggregation == "sum":
            val = getattr(proj, self.derived_field or "ytd_savings", 0)
            return float(val) if val is not None else None
        if self.aggregation == "weighted_avg":
            entries = proj.monthly_actuals or []
            total_weighted = 0.0
            total_volume = 0.0
            for entry in entries:
                actual = entry.get("actual")
                volume = entry.get("volume")
                if actual is not None and volume:
                    total_weighted += float(actual) * float(volume)
                    total_volume += float(volume)
            if total_volume > 0:
                return round(total_weighted / total_volume, 4)
            return None
        if self.aggregation == "latest":
            if self.calculator_result_type and hasattr(proj, "project_id"):
                try:
                    from agents_api.models import DSWResult

                    result = (
                        DSWResult.objects.filter(
                            user=proj.project.user,
                            result_type=self.calculator_result_type,
                            project=proj.project,
                        )
                        .order_by("-created_at")
                        .first()
                    )
                    if result and result.data:
                        import json as _json

                        data = _json.loads(result.data) if isinstance(result.data, str) else result.data
                        field = self.calculator_field or "cpk"
                        if field in data:
                            return float(data[field])
                        stats = data.get("statistics", {})
                        if field in stats:
                            return float(stats[field])
                except Exception:
                    pass
            return float(self.actual_value) if self.actual_value is not None else None
        return float(self.actual_value) if self.actual_value is not None else None

    def __str__(self):
        return self.name

    def to_dict(self):
        from agents_api.models import HoshinKPI as OrigKPI

        metric_type = self.calculator_result_type or "manual"
        meta = OrigKPI.METRIC_CATALOG.get(metric_type, {})
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "fiscal_year": self.fiscal_year,
            "name": self.name,
            "description": self.description,
            "target_value": float(self.target_value) if self.target_value else None,
            "actual_value": self.effective_actual,
            "unit": self.unit,
            "frequency": self.frequency,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "metric_type": metric_type,
            "metric_label": meta.get("label", "Manual Entry"),
            "derived_from_id": (str(self.derived_from_id) if self.derived_from_id else None),
            "derived_field": self.derived_field,
            "calculator_result_type": self.calculator_result_type,
            "calculator_field": self.calculator_field,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class XMatrixCorrelation(models.Model):
    """Relationship strength between X-matrix items."""

    class Strength(models.TextChoices):
        STRONG = "strong", "Strong"
        MODERATE = "moderate", "Moderate"
        WEAK = "weak", "Weak"

    class Source(models.TextChoices):
        AUTO = "auto", "Auto-suggested"
        MANUAL = "manual", "Manual"

    PAIR_CHOICES = [
        ("strategic_annual", "Strategic <-> Annual"),
        ("annual_project", "Annual <-> Project"),
        ("project_kpi", "Project <-> KPI"),
        ("kpi_strategic", "KPI <-> Strategic"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey("core.Tenant", on_delete=models.CASCADE, related_name="hoshin_xmatrix_correlations")
    fiscal_year = models.IntegerField(default=_current_year)
    pair_type = models.CharField(max_length=30, choices=PAIR_CHOICES)
    row_id = models.UUIDField()
    col_id = models.UUIDField()
    strength = models.CharField(max_length=10, choices=Strength.choices)
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.MANUAL)
    is_confirmed = models.BooleanField(default=False, db_column="confirmed")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_xmatrix_correlations"
        managed = False

    def __str__(self):
        return f"{self.pair_type}: {self.row_id} <-> {self.col_id} ({self.strength})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "pair_type": self.pair_type,
            "row_id": str(self.row_id),
            "col_id": str(self.col_id),
            "strength": self.strength,
            "source": self.source,
            "confirmed": self.is_confirmed,
        }

"""HIRARC Safety System models.

Implements: Frontier Card observations, audit scheduling, 5S assessment,
card-to-FMEA pipeline, and safety KPI tracking.

Models aligned to the printed Frontier Card specification:
- Front: 19 safety observation items across 6 categories (S/AR/U rating)
- Back: 26 5S deficiency items across 5 pillars (tally mode + 1-5 scoring)
- Operator interaction with comfort level
- Close-the-loop tracking

Standard: SAF-001, HIRARC-STD-001
"""

import uuid

from django.conf import settings
from django.db import models

# =============================================================================
# FRONTIER ZONES
# =============================================================================


ZONE_TYPE_CHOICES = [
    ("transition", "Transition Zone"),
    ("hidden_infra", "Hidden Infrastructure"),
    ("overhead_below", "Overhead & Below"),
    ("temporal", "Temporal Frontier"),
    ("general", "General"),
]

RISK_LEVEL_CHOICES = [
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("critical", "Critical"),
]

AUDIT_FREQUENCY_CHOICES = [
    ("weekly", "Weekly"),
    ("biweekly", "Biweekly"),
    ("monthly", "Monthly"),
    ("quarterly", "Quarterly"),
]


class FrontierZone(models.Model):
    """A defined area targeted for Frontier Card audits.

    Zones are deliberately non-standard: transition areas, overhead,
    below-floor, temporal frontiers (shift changes), hidden infrastructure.

    Zone categories from the Frontier Zone Selector (SAF-001 §6.1):
    - Transition: dept handoffs, loading dock edges, hallways, stairwells, exits, parking
    - Hidden Infrastructure: electrical/MCC, utility chases, behind equipment, maintenance cribs, chem storage
    - Overhead & Below: mezzanine/catwalks, under conveyors, above ceilings, pits, roof access, cable trays
    - Temporal: shift changeover, seasonal storage, temporary staging, break rooms, training areas, contractor zones

    Metadata fields support automation:
    - Risk profile → audit frequency, FMEA severity seeding, hazard-matched card items
    - 5S baseline/target → trend tracking, red/green status per zone
    - Scheduling → frequency, last_audited, preferred auditors
    - Physical context → location detail, photo reference, zone hierarchy
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.CASCADE,
        related_name="frontier_zones",
    )
    name = models.CharField(max_length=255, help_text="e.g., 'Press mezzanine', 'Loading dock transition'")
    description = models.TextField(blank=True, default="")
    zone_type = models.CharField(max_length=30, choices=ZONE_TYPE_CHOICES, default="general")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # --- Risk Profile ---
    risk_level = models.CharField(
        max_length=20,
        choices=RISK_LEVEL_CHOICES,
        default="medium",
        help_text="Baseline HIRARC risk. Drives audit frequency and FMEA severity seeding.",
    )
    hazard_types = models.JSONField(
        default=list,
        blank=True,
        help_text="e.g. ['electrical', 'fall', 'confined_space']. Auto-surfaces relevant card items.",
    )
    hierarchy_controls = models.JSONField(
        default=dict,
        blank=True,
        help_text="Controls in place per hierarchy level: {elimination: '', substitution: '', engineering: '', administrative: '', ppe: ''}",
    )

    # --- 5S Baseline ---
    five_s_baseline = models.JSONField(
        default=dict,
        blank=True,
        help_text="Initial 5S scores: {sort: 1-5, set_in_order: 1-5, shine: 1-5, standardize: 1-5, sustain: 1-5}",
    )
    five_s_target = models.JSONField(
        default=dict,
        blank=True,
        help_text="Target 5S scores per pillar. Enables delta tracking and red/green status.",
    )

    # --- Scheduling Intelligence ---
    audit_frequency = models.CharField(
        max_length=20,
        choices=AUDIT_FREQUENCY_CHOICES,
        default="weekly",
        help_text="How often this zone should appear in audit schedules.",
    )
    last_audited = models.DateField(
        null=True,
        blank=True,
        help_text="Denormalized from latest card. Enables overdue-zone queries.",
    )
    preferred_auditors = models.JSONField(
        default=list,
        blank=True,
        help_text="Employee UUIDs of auditors who know this zone best.",
    )

    # --- Physical Context ---
    location_detail = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Building/floor/area code. e.g. 'Bldg 3, 2nd floor, east wing'",
    )
    photo_reference = models.URLField(
        blank=True,
        default="",
        help_text="Photo URL showing normal/baseline condition of the zone.",
    )
    parent_zone = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="sub_zones",
        help_text="Parent zone for hierarchy rollup. e.g. mezzanine → press area.",
    )

    # --- Automation Hooks ---
    auto_fmea_severity = models.IntegerField(
        null=True,
        blank=True,
        help_text="Override default FMEA severity (5) for cards from this zone. Critical zones seed at 8-10.",
    )
    tags = models.JSONField(
        default=list,
        blank=True,
        help_text="Freeform tags for filtering/grouping. e.g. ['new_process', 'contractor', 'night_shift']",
    )

    class Meta:
        db_table = "safety_frontier_zones"
        ordering = ["site", "name"]

    def __str__(self):
        return f"{self.name} ({self.site.name})"

    @property
    def is_overdue(self):
        """True if zone hasn't been audited within its frequency window."""
        if not self.last_audited:
            return True
        from datetime import date

        freq_days = {"weekly": 7, "biweekly": 14, "monthly": 30, "quarterly": 90}
        window = freq_days.get(self.audit_frequency, 7)
        return (date.today() - self.last_audited).days > window

    def to_dict(self):
        return {
            "id": str(self.id),
            "site_id": str(self.site_id),
            "name": self.name,
            "description": self.description,
            "zone_type": self.zone_type,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "risk_level": self.risk_level,
            "hazard_types": self.hazard_types,
            "hierarchy_controls": self.hierarchy_controls,
            "five_s_baseline": self.five_s_baseline,
            "five_s_target": self.five_s_target,
            "audit_frequency": self.audit_frequency,
            "last_audited": (self.last_audited.isoformat() if self.last_audited else None),
            "preferred_auditors": self.preferred_auditors,
            "location_detail": self.location_detail,
            "photo_reference": self.photo_reference,
            "parent_zone_id": str(self.parent_zone_id) if self.parent_zone_id else None,
            "auto_fmea_severity": self.auto_fmea_severity,
            "tags": self.tags,
            "is_overdue": self.is_overdue,
        }


# =============================================================================
# AUDIT SCHEDULING
# =============================================================================


class AuditSchedule(models.Model):
    """Weekly audit schedule — assigns auditors to frontier zones.

    Published by Area Manager every Monday. Tracked for completion.
    SAF-001 §6.3: audits are frequent, planned, assigned — not voluntary.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.CASCADE,
        related_name="audit_schedules",
    )
    week_start = models.DateField(help_text="Monday of the audit week")
    published_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="published_audit_schedules",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "safety_audit_schedules"
        unique_together = [("site", "week_start")]
        ordering = ["-week_start"]

    def __str__(self):
        return f"Audit schedule: {self.site.name} week of {self.week_start}"

    @property
    def completion_rate(self):
        total = self.assignments.count()
        if not total:
            return 0
        completed = self.assignments.filter(status="completed").count()
        return round(completed / total * 100, 1)


class AuditAssignment(models.Model):
    """Individual audit assignment — auditor + zone + target date."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    schedule = models.ForeignKey(
        AuditSchedule,
        on_delete=models.CASCADE,
        related_name="assignments",
    )
    auditor = models.ForeignKey(
        "agents_api.Employee",
        on_delete=models.CASCADE,
        related_name="audit_assignments",
    )
    zone = models.ForeignKey(
        FrontierZone,
        on_delete=models.CASCADE,
        related_name="assignments",
    )
    target_date = models.DateField()
    status = models.CharField(
        max_length=20,
        choices=[
            ("assigned", "Assigned"),
            ("in_progress", "In Progress"),
            ("completed", "Completed"),
            ("missed", "Missed"),
        ],
        default="assigned",
    )
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "safety_audit_assignments"
        ordering = ["target_date"]

    def __str__(self):
        return f"{self.auditor.name} → {self.zone.name} ({self.target_date})"


# =============================================================================
# FRONTIER CARD
# =============================================================================


class FrontierCard(models.Model):
    """A completed Frontier Card observation — the primary HI vector.

    Matches the printed Frontier Card specification exactly:

    FRONT (Safety): 19 observation items across 6 categories.
    Each item rated S (Satisfactory), AR (At Risk), or U (Unsatisfactory).
    At-risk/unsatisfactory findings get severity tag: C/H/M/L.

    BACK (5S): 26 deficiency items across 5 pillars.
    Two modes: tally (quick card) or 1-5 scoring (detailed audit).

    Operator interaction: 3 required questions + comfort level.
    Close-the-loop: documented feedback to operator.

    SAF-001 §5
    """

    class Severity(models.TextChoices):
        CRITICAL = "C", "Critical (9-10)"
        HIGH = "H", "High (7-8)"
        MEDIUM = "M", "Medium (4-6)"
        LOW = "L", "Low (1-3)"

    class AuditClassification(models.TextChoices):
        ROUTINE = "routine", "Routine Frontier"
        DEEP_DIVE = "deep_dive", "Deep Dive"
        POST_INCIDENT = "post_incident", "Post-Incident"
        NEW_PROCESS = "new_process", "New Process/Area"
        SEASONAL = "seasonal", "Seasonal/Temporary"

    class CloseLoopMethod(models.TextChoices):
        IMMEDIATE = "immediate", "Yes, on the spot"
        WITHIN_24H = "within_24h", "Within 24 hours"
        PENDING = "pending", "Pending"
        NOT_DONE = "not_done", "Not done"

    SEVERITY_TO_FMEA = {"C": 10, "H": 8, "M": 5, "L": 2}

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    assignment = models.ForeignKey(
        AuditAssignment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="cards",
        help_text="Link to scheduled assignment (null if ad-hoc)",
    )
    auditor = models.ForeignKey(
        "agents_api.Employee",
        on_delete=models.CASCADE,
        related_name="frontier_cards",
    )
    zone = models.ForeignKey(
        FrontierZone,
        on_delete=models.CASCADE,
        related_name="cards",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.CASCADE,
        related_name="frontier_cards",
    )
    audit_date = models.DateField()
    shift = models.CharField(max_length=20, blank=True, default="")
    classification = models.CharField(
        max_length=20,
        choices=AuditClassification.choices,
        default=AuditClassification.ROUTINE,
        help_text="Type of frontier audit",
    )
    zone_reason = models.CharField(
        max_length=300,
        blank=True,
        default="",
        help_text="Why this zone? What made you suspicious?",
    )

    # ── Front: Safety Observations (S/AR/U rating) ──
    # [{category, item, rating: "S"|"AR"|"U", severity: "C"|"H"|"M"|"L"|null, notes}]
    safety_observations = models.JSONField(
        default=list,
        help_text='[{category, item, rating: "S"|"AR"|"U", severity, notes}]',
    )

    # ── Back: 5S Assessment ──
    # Tally mode (quick card): {pillar: deficiency_count}
    five_s_tallies = models.JSONField(
        default=dict,
        help_text="{sort, set_in_order, shine, standardize, sustain} — deficiency counts",
    )
    # Detailed mode (full audit): [{pillar, item, score: 1-5, notes}]
    five_s_scores = models.JSONField(
        default=list,
        help_text="[{pillar, item, score: 1-5, notes}] — detailed 5S assessment",
    )

    # ── Positive Observations ──
    positive_observations = models.JSONField(
        default=list,
        help_text="[{observation, reinforce_how}] — safe behaviors to recognize",
    )

    # ── Operator Interaction ──
    operator_name = models.CharField(max_length=255, blank=True, default="")
    operator_role = models.CharField(max_length=100, blank=True, default="")
    operator_tenure = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="How long in this area",
    )
    # Three required questions (from card)
    operator_concern = models.TextField(
        blank=True,
        default="",
        help_text="What's the most hazardous thing about working here?",
    )
    operator_ergonomic = models.TextField(
        blank=True,
        default="",
        help_text="What part of your body is most tired at end of shift?",
    )
    operator_improvement = models.TextField(
        blank=True,
        default="",
        help_text="If you could change one thing here, what would it be?",
    )
    operator_near_miss = models.TextField(
        blank=True,
        default="",
        help_text="Has anyone had a close call here recently?",
    )
    operator_comfort_level = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text="1=Reluctant/guarded, 3=Cooperative, 5=Enthusiastic/engaged",
    )
    operator_behavior_notes = models.TextField(
        blank=True,
        default="",
        help_text="Auditor's observation of operator behavior during task",
    )
    operator_five_s_issue = models.CharField(
        max_length=300,
        blank=True,
        default="",
        help_text="Operator's #1 5S issue in the area",
    )
    operator_quick_win = models.CharField(
        max_length=300,
        blank=True,
        default="",
        help_text="Quick win spotted during conversation",
    )

    # ── Close the Loop ──
    close_loop_method = models.CharField(
        max_length=20,
        choices=CloseLoopMethod.choices,
        default=CloseLoopMethod.PENDING,
        help_text="How feedback was delivered to operator",
    )
    close_loop_notes = models.TextField(
        blank=True,
        default="",
        help_text="Operator's response to feedback",
    )
    close_loop_followup_date = models.DateField(
        null=True,
        blank=True,
        help_text="Scheduled follow-up date",
    )

    # ── Processing status ──
    is_processed = models.BooleanField(
        default=False,
        help_text="True after card findings have been entered into FMEA",
    )
    processed_at = models.DateTimeField(null=True, blank=True)
    processed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="processed_cards",
    )

    # ── 5S cross-feed ──
    has_safety_crossfeed = models.BooleanField(
        default=False,
        help_text="True if any 5S deficiency also carries safety consequence",
    )
    crossfeed_notes = models.TextField(blank=True, default="")

    # ── Linked FMEA rows ──
    fmea_rows_created = models.JSONField(
        default=list,
        help_text="UUIDs of FMEARow records created from this card",
    )

    # ── Data pipeline checklist (from card back) ──
    pipeline_logged = models.BooleanField(default=False, help_text="Card logged to tracking")
    pipeline_tallies_entered = models.BooleanField(default=False, help_text="5S tallies entered for Pareto")
    pipeline_safety_to_fmea = models.BooleanField(default=False, help_text="Safety findings entered to FMEA")
    pipeline_feedback_given = models.BooleanField(default=False, help_text="Feedback given to operator")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "safety_frontier_cards"
        ordering = ["-audit_date", "-created_at"]
        indexes = [
            models.Index(fields=["site", "-audit_date"]),
            models.Index(fields=["auditor", "-audit_date"]),
            models.Index(fields=["is_processed"]),
        ]

    def __str__(self):
        return f"Card: {self.zone.name} by {self.auditor.name} ({self.audit_date})"

    @property
    def at_risk_count(self):
        """Count observations rated AR (At Risk) or U (Unsatisfactory)."""
        return sum(1 for o in (self.safety_observations or []) if o.get("rating") in ("AR", "U"))

    @property
    def satisfactory_count(self):
        return sum(1 for o in (self.safety_observations or []) if o.get("rating") == "S")

    @property
    def highest_severity(self):
        sevs = [
            o.get("severity", "L")
            for o in (self.safety_observations or [])
            if o.get("rating") in ("AR", "U") and o.get("severity")
        ]
        if not sevs:
            return None
        order = {"C": 0, "H": 1, "M": 2, "L": 3}
        return min(sevs, key=lambda s: order.get(s, 99))

    @property
    def total_five_s_deficiencies(self):
        return sum((self.five_s_tallies or {}).values())

    @property
    def five_s_avg_score(self):
        """Average 1-5 score from detailed 5S assessment."""
        scores = [s.get("score", 0) for s in (self.five_s_scores or []) if s.get("score")]
        return round(sum(scores) / len(scores), 1) if scores else None

    @property
    def satisfactory_pct(self):
        """Percentage of safety observations rated Satisfactory."""
        total = len(self.safety_observations or [])
        if not total:
            return None
        return round(self.satisfactory_count / total * 100, 1)

    def severity_to_fmea_score(self, severity_tag):
        """Map card severity tag to FMEA severity score (SAF-001 §4.3.2)."""
        return self.SEVERITY_TO_FMEA.get(severity_tag, 2)

    def to_dict(self):
        return {
            "id": str(self.id),
            "auditor_id": str(self.auditor_id),
            "auditor_name": self.auditor.name,
            "zone_id": str(self.zone_id),
            "zone_name": self.zone.name,
            "site_id": str(self.site_id),
            "audit_date": self.audit_date.isoformat(),
            "shift": self.shift,
            "classification": self.classification,
            "zone_reason": self.zone_reason,
            "safety_observations": self.safety_observations,
            "five_s_tallies": self.five_s_tallies,
            "five_s_scores": self.five_s_scores,
            "positive_observations": self.positive_observations,
            "operator_name": self.operator_name,
            "operator_role": self.operator_role,
            "operator_concern": self.operator_concern,
            "operator_ergonomic": self.operator_ergonomic,
            "operator_improvement": self.operator_improvement,
            "operator_near_miss": self.operator_near_miss,
            "operator_comfort_level": self.operator_comfort_level,
            "operator_five_s_issue": self.operator_five_s_issue,
            "operator_quick_win": self.operator_quick_win,
            "close_loop_method": self.close_loop_method,
            "at_risk_count": self.at_risk_count,
            "satisfactory_pct": self.satisfactory_pct,
            "highest_severity": self.highest_severity,
            "total_five_s_deficiencies": self.total_five_s_deficiencies,
            "five_s_avg_score": self.five_s_avg_score,
            "is_processed": self.is_processed,
            "has_safety_crossfeed": self.has_safety_crossfeed,
            "fmea_rows_created": self.fmea_rows_created,
            "pipeline_logged": self.pipeline_logged,
            "pipeline_tallies_entered": self.pipeline_tallies_entered,
            "pipeline_safety_to_fmea": self.pipeline_safety_to_fmea,
            "pipeline_feedback_given": self.pipeline_feedback_given,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# SAFETY OBSERVATION CATEGORIES — Exact card items
# =============================================================================

# Front of card: 19 items across 6 categories. Each rated S/AR/U.
SAFETY_OBSERVATION_CATEGORIES = {
    "body_position": {
        "label": "Body Position & Ergonomics",
        "items": [
            "Lifting posture (load close, knees bent)",
            "Reaching above shoulder / below knee",
            "Repetitive motion or awkward sustained posture",
            "Workstation height matches task",
        ],
    },
    "ppe": {
        "label": "PPE Compliance",
        "items": [
            "Correct PPE for zone/task, good condition",
            "PPE available at point of use",
        ],
    },
    "surfaces": {
        "label": "Walking & Working Surfaces",
        "items": [
            "Trip/slip hazards (cords, spills, debris)",
            "Guardrails/handrails present & secure",
            "Floor markings visible & accurate",
        ],
    },
    "energy": {
        "label": "Energy & Hazardous Materials",
        "items": [
            "LOTO applied where required",
            'Electrical panels: 36" clearance',
            "Chemical containers labeled, contained",
            "Compressed gas chained/capped",
        ],
    },
    "emergency": {
        "label": "Emergency Preparedness",
        "items": [
            "Extinguisher/eyewash accessible",
            "Exit paths clear & marked",
            "Spill/first aid kit stocked",
        ],
    },
    "environmental": {
        "label": "Environmental Conditions",
        "items": [
            "Lighting adequate for task",
            "Ventilation/air quality acceptable",
            "Noise: can converse at arm's length?",
        ],
    },
}


# Back of card: 26 items across 5 pillars
FIVE_S_PILLARS = {
    "sort": {
        "label": "Sort (Seiri)",
        "items": [
            "Unnecessary items present",
            "Red-tag candidates (unused/obsolete)",
            '"Just in case" hoarding',
            "Personal items outside designated area",
            "Broken/expired items not removed",
        ],
    },
    "set_in_order": {
        "label": "Set in Order (Seiton)",
        "items": [
            "No designated location / no label",
            "Shadow board missing or incomplete",
            "No FIFO / no date on consumables",
            "Frequently used item not at ergo height",
            "No visual min/max indicator",
            "Return path unclear (where does this go?)",
        ],
    },
    "shine": {
        "label": "Shine (Seiso)",
        "items": [
            "Visible dirt/grime/buildup on equipment",
            "No cleaning schedule posted",
            "Drains/vents/filters blocked or dirty",
            "Evidence of pest activity",
            "Inspection points obscured",
        ],
    },
    "standardize": {
        "label": "Standardize (Seiketsu)",
        "items": [
            "No visual work instruction at point of use",
            "Color coding inconsistent with facility std",
            'No "standard condition" photo/reference',
            "Zone ownership unclear (who owns this?)",
            "Can't spot abnormality at a glance",
        ],
    },
    "sustain": {
        "label": "Sustain (Shitsuke)",
        "items": [
            "No evidence of prior audits/checks",
            "Operator can't explain 5S standard here",
            "No visible improvement tracking",
            "Zone doesn't recover after disruption",
        ],
    },
}


# =============================================================================
# CARD-TO-FMEA PIPELINE
# =============================================================================


def process_card_to_fmea(card, fmea, user):
    """Create FMEA rows from a Frontier Card's at-risk/unsatisfactory findings.

    SAF-001 §4.3.3: Hazard → Risk transformation via FMEA scoring.
    Only AR (At Risk) and U (Unsatisfactory) rated observations generate FMEA rows.

    Args:
        card: FrontierCard instance with safety_observations
        fmea: agents_api.models.FMEA instance to add rows to
        user: User performing the processing

    Returns:
        list of created FMEARow IDs
    """
    from django.utils import timezone

    from agents_api.models import FMEARow

    created_ids = []
    for obs in card.safety_observations or []:
        rating = obs.get("rating", "S")
        if rating not in ("AR", "U"):
            continue

        severity = card.severity_to_fmea_score(obs.get("severity", "L"))
        category = obs.get("category", "")
        item = obs.get("item", "")
        notes = obs.get("notes", "")

        row = FMEARow.objects.create(
            fmea=fmea,
            process_step=f"[{card.zone.name}] {category}",
            failure_mode=item,
            effect=notes
            or f"{'Unsatisfactory' if rating == 'U' else 'At-risk'} condition observed during Frontier audit",
            severity=severity,
            occurrence=5,  # Default — refined during weekly FMEA review
            detection=5,  # Default — refined during weekly FMEA review
            current_controls=f"Frontier Card audit ({card.audit_date})",
            recommended_action="Review and assign control per hierarchy (SAF-001 §4.4)",
            action_owner=card.auditor.name,
        )
        created_ids.append(str(row.id))

    # Process 5S cross-feed items
    if card.has_safety_crossfeed and card.crossfeed_notes:
        row = FMEARow.objects.create(
            fmea=fmea,
            process_step=f"[{card.zone.name}] 5S Cross-feed",
            failure_mode="5S deficiency with safety consequence",
            effect=card.crossfeed_notes,
            severity=6,  # Medium default for cross-feed
            occurrence=5,
            detection=5,
            current_controls="5S audit + cross-feed to FMEA",
            recommended_action="Address 5S root cause + implement safety control",
            action_owner=card.auditor.name,
        )
        created_ids.append(str(row.id))

    # Mark card as processed
    card.is_processed = True
    card.processed_at = timezone.now()
    card.processed_by = user
    card.fmea_rows_created = created_ids
    card.pipeline_safety_to_fmea = True
    card.save(
        update_fields=[
            "is_processed",
            "processed_at",
            "processed_by",
            "fmea_rows_created",
            "pipeline_safety_to_fmea",
            "updated_at",
        ]
    )

    return created_ids


# =============================================================================
# 5S PARETO AGGREGATION
# =============================================================================


def aggregate_five_s_pareto(site, min_cards=10):
    """Aggregate 5S tallies from Frontier Cards into Pareto data.

    SAF-001 §4.5: Stack-rank by frequency for standardization.
    Minimum 10 audits required before generating Pareto.

    Args:
        site: Site instance
        min_cards: Minimum cards required for meaningful Pareto

    Returns:
        dict with pareto data and total card count, or None if insufficient data
    """
    cards = FrontierCard.objects.filter(site=site).exclude(five_s_tallies={})
    count = cards.count()
    if count < min_cards:
        return None

    # Aggregate tallies across all cards
    totals = {}
    for card in cards:
        for pillar, tally in (card.five_s_tallies or {}).items():
            totals[pillar] = totals.get(pillar, 0) + (tally or 0)

    if not totals:
        return None

    # Sort by frequency (descending) for Pareto
    grand_total = sum(totals.values())
    sorted_items = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    pareto = []
    cumulative = 0
    for pillar, count_val in sorted_items:
        cumulative += count_val
        pillar_info = FIVE_S_PILLARS.get(pillar, {})
        pareto.append(
            {
                "pillar": pillar,
                "label": pillar_info.get("label", pillar),
                "count": count_val,
                "pct": round(count_val / grand_total * 100, 1) if grand_total else 0,
                "cumulative_pct": (round(cumulative / grand_total * 100, 1) if grand_total else 0),
            }
        )

    return {
        "site_id": str(site.id),
        "card_count": count,
        "grand_total": grand_total,
        "pareto": pareto,
    }

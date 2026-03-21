"""HIRARC Safety System models.

Implements: Frontier Card observations, audit scheduling, 5S tallies,
card-to-FMEA pipeline, and safety KPI tracking.

Standard: HIRARC-STD-001
"""

import uuid

from django.conf import settings
from django.db import models

# =============================================================================
# FRONTIER ZONES
# =============================================================================


class FrontierZone(models.Model):
    """A defined area targeted for Frontier Card audits.

    Zones are deliberately non-standard: transition areas, overhead,
    below-floor, temporal frontiers (shift changes), hidden infrastructure.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.CASCADE,
        related_name="frontier_zones",
    )
    name = models.CharField(max_length=255, help_text="e.g., 'Press mezzanine', 'Loading dock transition'")
    description = models.TextField(blank=True, default="")
    zone_type = models.CharField(
        max_length=30,
        choices=[
            ("transition", "Transition Zone"),
            ("overhead", "Overhead / Elevated"),
            ("below", "Below Floor / Pit"),
            ("infrastructure", "Hidden Infrastructure"),
            ("temporal", "Temporal (Shift Change)"),
            ("contractor", "Contractor Work Zone"),
            ("storage", "Storage / Warehouse"),
            ("general", "General"),
        ],
        default="general",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "safety_frontier_zones"
        ordering = ["site", "name"]

    def __str__(self):
        return f"{self.name} ({self.site.name})"


# =============================================================================
# AUDIT SCHEDULING
# =============================================================================


class AuditSchedule(models.Model):
    """Weekly audit schedule — assigns auditors to frontier zones.

    Published by Area Manager every Monday. Tracked for completion.
    HIRARC-STD-001 §5.1: audits are frequent, planned, assigned — not voluntary.
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

    Front: 20 safety observation items across 6 categories.
    Back: 25 5S deficiency tallies across 5 pillars.
    Interaction: 3 operator questions.

    HIRARC-STD-001 §4.1, §5.2
    """

    class Severity(models.TextChoices):
        CRITICAL = "C", "Critical (9-10)"
        HIGH = "H", "High (7-8)"
        MEDIUM = "M", "Medium (4-6)"
        LOW = "L", "Low (1-3)"

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

    # ── Front: Safety Observations ──
    # JSON array of findings: [{category, item, at_risk: bool, severity: C/H/M/L, notes}]
    safety_observations = models.JSONField(
        default=list,
        help_text="[{category, item, at_risk, severity, notes}]",
    )

    # ── Back: 5S Tallies ──
    # JSON object: {sort: int, set_in_order: int, shine: int, standardize: int, sustain: int}
    five_s_tallies = models.JSONField(
        default=dict,
        help_text="{sort, set_in_order, shine, standardize, sustain} — deficiency counts",
    )

    # ── Operator Interaction ──
    operator_name = models.CharField(max_length=255, blank=True, default="")
    operator_concern = models.TextField(
        blank=True,
        default="",
        help_text="What is your biggest safety concern in this area?",
    )
    operator_improvement = models.TextField(
        blank=True,
        default="",
        help_text="If you could change one thing to make this area safer, what would it be?",
    )
    operator_near_miss = models.TextField(
        blank=True,
        default="",
        help_text="Have you experienced or witnessed any near-misses here recently?",
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
        return sum(1 for o in (self.safety_observations or []) if o.get("at_risk"))

    @property
    def highest_severity(self):
        sevs = [o.get("severity", "L") for o in (self.safety_observations or []) if o.get("at_risk")]
        if not sevs:
            return None
        order = {"C": 0, "H": 1, "M": 2, "L": 3}
        return min(sevs, key=lambda s: order.get(s, 99))

    @property
    def total_five_s_deficiencies(self):
        return sum((self.five_s_tallies or {}).values())

    def severity_to_fmea_score(self, severity_tag):
        """Map card severity tag to FMEA severity score (HIRARC-STD-001 §4.2.2)."""
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
            "safety_observations": self.safety_observations,
            "five_s_tallies": self.five_s_tallies,
            "operator_name": self.operator_name,
            "operator_concern": self.operator_concern,
            "operator_improvement": self.operator_improvement,
            "operator_near_miss": self.operator_near_miss,
            "at_risk_count": self.at_risk_count,
            "highest_severity": self.highest_severity,
            "total_five_s_deficiencies": self.total_five_s_deficiencies,
            "is_processed": self.is_processed,
            "has_safety_crossfeed": self.has_safety_crossfeed,
            "fmea_rows_created": self.fmea_rows_created,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# SAFETY OBSERVATION CATEGORIES (Reference Data)
# =============================================================================

# HIRARC-STD-001 §5.2 — 20 items across 6 categories for the Frontier Card front
SAFETY_OBSERVATION_CATEGORIES = {
    "body_position": {
        "label": "Body Position & Ergonomics",
        "items": [
            "Lifting technique (back straight, knees bent)",
            "Repetitive motion / awkward posture",
            "Working at height without protection",
            "Pinch points / caught-between exposure",
        ],
    },
    "ppe": {
        "label": "Personal Protective Equipment",
        "items": [
            "Required PPE present and worn correctly",
            "PPE condition (damaged, expired, improper fit)",
            "PPE appropriate for the hazard",
        ],
    },
    "housekeeping": {
        "label": "Housekeeping & Organization",
        "items": [
            "Walking surfaces clear and dry",
            "Materials stored properly (no overhead hazards)",
            "Waste / debris managed",
            "Emergency exits / paths unobstructed",
        ],
    },
    "energy_control": {
        "label": "Energy Control & Guarding",
        "items": [
            "LOTO procedures followed",
            "Machine guarding in place and functional",
            "Electrical panels accessible / properly marked",
            "Compressed gas cylinders secured",
        ],
    },
    "chemical_environmental": {
        "label": "Chemical & Environmental",
        "items": [
            "SDS available for chemicals in use",
            "Chemical containers labeled",
            "Ventilation adequate",
            "Noise levels acceptable / hearing protection available",
        ],
    },
    "tools_equipment": {
        "label": "Tools & Equipment",
        "items": [
            "Tools in good condition (no jury-rigging)",
            "Equipment inspection current",
        ],
    },
}

# 5S pillar items for card back (25 total, 5 per pillar)
FIVE_S_PILLARS = {
    "sort": {
        "label": "Sort (Seiri)",
        "items": [
            "Unnecessary items present",
            "Red tag items not dispositioned",
            "Personal items in work area",
            "Obsolete tools / materials",
            "Excess inventory beyond kanban",
        ],
    },
    "set_in_order": {
        "label": "Set in Order (Seiton)",
        "items": [
            "Items not in designated locations",
            "Shadow boards incomplete",
            "Labels missing or illegible",
            "Visual controls absent",
            "FIFO not maintained",
        ],
    },
    "shine": {
        "label": "Shine (Seiso)",
        "items": [
            "Equipment surfaces dirty",
            "Fluid leaks present",
            "Lighting inadequate",
            "Floors not clean",
            "Inspection points obscured",
        ],
    },
    "standardize": {
        "label": "Standardize (Seiketsu)",
        "items": [
            "SOPs not posted or outdated",
            "Color coding inconsistent",
            "Cleaning schedule not followed",
            "Zone boundaries unclear",
            "Audit results not posted",
        ],
    },
    "sustain": {
        "label": "Sustain (Shitsuke)",
        "items": [
            "Previous audit findings not addressed",
            "Standards drifting from baseline",
            "Team engagement declining",
            "Recognition absent",
            "Improvement suggestions stale",
        ],
    },
}


# =============================================================================
# CARD-TO-FMEA PIPELINE
# =============================================================================


def process_card_to_fmea(card, fmea, user):
    """Create FMEA rows from a Frontier Card's at-risk findings.

    HIRARC-STD-001 §4.2.1: Hazard → Risk transformation via FMEA scoring.

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
        if not obs.get("at_risk"):
            continue

        severity = card.severity_to_fmea_score(obs.get("severity", "L"))
        category = obs.get("category", "")
        item = obs.get("item", "")
        notes = obs.get("notes", "")

        row = FMEARow.objects.create(
            fmea=fmea,
            process_step=f"[{card.zone.name}] {category}",
            failure_mode=item,
            effect=notes or "At-risk condition observed during Frontier audit",
            severity=severity,
            occurrence=5,  # Default — refined during weekly FMEA review
            detection=5,  # Default — refined during weekly FMEA review
            current_controls=f"Frontier Card audit ({card.audit_date})",
            recommended_action="Review and assign control per hierarchy (HIRARC-STD-001 §4.2.3)",
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
    card.save(
        update_fields=[
            "is_processed",
            "processed_at",
            "processed_by",
            "fmea_rows_created",
            "updated_at",
        ]
    )

    return created_ids


# =============================================================================
# 5S PARETO AGGREGATION
# =============================================================================


def aggregate_five_s_pareto(site, min_cards=10):
    """Aggregate 5S tallies from Frontier Cards into Pareto data.

    HIRARC-STD-001 §4.3: Stack-rank by frequency for standardization.
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
                "cumulative_pct": round(cumulative / grand_total * 100, 1) if grand_total else 0,
            }
        )

    return {
        "site_id": str(site.id),
        "card_count": count,
        "grand_total": grand_total,
        "pareto": pareto,
    }

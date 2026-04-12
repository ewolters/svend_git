"""
Configuration Service — OLR-001 §16, enterprise_configuration_spec.md.

One table, one service. Every configurable behavior in the platform
reads from here. TenantConfig model in graph/models.py.
ConfigService provides get/set with preset support.
"""

import logging
from uuid import UUID

from .models import TenantConfig

logger = logging.getLogger("svend.graph.config")


# =============================================================================
# Defaults + Presets
# =============================================================================

DEFAULTS = {
    # Quality
    "quality.ncr.require_root_cause": True,
    "quality.ncr.require_containment": False,
    "quality.ncr.auto_create_investigation": False,
    "quality.ncr.escalation_days": 14,
    "quality.ncr.require_verification": True,
    "quality.capa.require_effectiveness_review": False,
    "quality.capa.effectiveness_review_days": 90,
    "quality.fmea.methodology": "aiag_4th",
    "quality.fmea.rpn_action_threshold": 100,
    "quality.fmea.severity_threshold": 8,
    "quality.fmea.require_detection_control": True,
    "quality.supplier.response_due_days": 14,
    "quality.supplier.auto_escalate_rejections": 2,
    "quality.supplier.quality_score_threshold": 0.5,
    "quality.audit.require_checklist": False,
    "quality.audit.finding_auto_ncr": False,
    "quality.audit.minimum_frequency_months": 12,
    "quality.training.require_renewal": True,
    "quality.training.default_renewal_months": 12,
    "quality.document.require_approval": True,
    "quality.document.review_reminder_days": 30,
    "quality.document.retention_years": 7,
    "quality.control_plan.require_reaction_plan": True,
    "quality.control_plan.require_gage_link": False,
    # Process
    "process.graph.evidence_decay_half_life_days": 180,
    "process.graph.contradiction_threshold": 0.05,
    "process.graph.contradiction_cooldown_days": 7,
    "process.graph.staleness_max_days": 365,
    "process.graph.auto_seed_from_fmis": False,
    "process.spc.auto_signal_on_ooc": False,
    "process.spc.signal_debounce_hours": 8,
    "process.spc.flag_stale_edges": True,
    "process.investigation.max_duration_days": 90,
    "process.investigation.writeback_on_conclude": True,
    "process.pc.require_photo": False,
    "process.fft.minimum_injection_count": 3,
    "process.fft.require_safety_review": True,
    # Classification staleness thresholds (OLR-001 §4.8)
    "process.staleness.critical_days": 90,
    "process.staleness.major_days": 180,
    "process.staleness.minor_days": 365,
    # Detection mechanism minimums (OLR-001 §9.3)
    "process.detection.critical_min_level": 4,
    "process.detection.major_min_level": 5,
    "process.detection.minor_min_level": 8,
    # Safety
    "safety.observation_target_per_week": 5,
    "safety.require_immediate_action": True,
    "safety.auto_fmea_from_card": True,
    # Compliance
    "compliance.standard": "iso_9001",
    "compliance.management_review_frequency_months": 12,
    "compliance.require_electronic_signatures": False,
    # Organization
    "org.company_name": "",
    "org.timezone": "UTC",
    "org.date_format": "iso",
}

PRESETS = {
    "iso_9001": {
        "compliance.standard": "iso_9001",
        "quality.ncr.require_root_cause": True,
        "quality.ncr.require_verification": True,
        "quality.fmea.methodology": "aiag_4th",
        "quality.audit.minimum_frequency_months": 12,
        "quality.document.require_approval": True,
        "quality.document.retention_years": 7,
        "quality.control_plan.require_reaction_plan": True,
    },
    "iatf_16949": {
        "compliance.standard": "iatf_16949",
        "quality.ncr.require_root_cause": True,
        "quality.ncr.require_containment": True,
        "quality.ncr.require_verification": True,
        "quality.fmea.require_detection_control": True,
        "quality.capa.require_effectiveness_review": True,
        "quality.capa.effectiveness_review_days": 90,
        "quality.audit.require_checklist": True,
        "quality.audit.finding_auto_ncr": True,
        "quality.control_plan.require_gage_link": True,
        "quality.supplier.auto_escalate_rejections": 2,
        "compliance.require_electronic_signatures": True,
        "process.fft.require_safety_review": True,
        "process.staleness.critical_days": 90,
        "process.staleness.major_days": 120,
    },
    "as9100d": {
        "compliance.standard": "as9100d",
        "quality.ncr.require_containment": True,
        "quality.ncr.escalation_days": 7,
        "quality.document.retention_years": 10,
        "compliance.require_electronic_signatures": True,
        "process.fft.minimum_injection_count": 5,
        "process.staleness.critical_days": 90,
        "process.staleness.major_days": 120,
        "process.staleness.minor_days": 365,
    },
    "lightweight": {
        "compliance.standard": "custom",
        "quality.ncr.require_root_cause": False,
        "quality.ncr.require_verification": False,
        "quality.fmea.rpn_action_threshold": 200,
        "quality.audit.minimum_frequency_months": 0,
        "quality.document.require_approval": False,
        "quality.training.require_renewal": False,
        "process.graph.staleness_max_days": 0,
    },
}


class ConfigService:
    """Read/write tenant configuration."""

    @staticmethod
    def get(tenant_id: UUID, key: str, site_id: UUID | None = None):
        """Get a config value. Site override → tenant default → hardcoded default."""
        if site_id:
            try:
                return TenantConfig.objects.get(tenant_id=tenant_id, key=key, site_id=site_id).value
            except TenantConfig.DoesNotExist:
                pass

        try:
            return TenantConfig.objects.get(tenant_id=tenant_id, key=key, site__isnull=True).value
        except TenantConfig.DoesNotExist:
            pass

        return DEFAULTS.get(key)

    @staticmethod
    def set(tenant_id: UUID, key: str, value, site_id: UUID | None = None, updated_by=None):
        """Set a config value."""
        domain = key.split(".")[0] if "." in key else "general"
        obj, created = TenantConfig.objects.update_or_create(
            tenant_id=tenant_id,
            domain=domain,
            key=key,
            site_id=site_id,
            defaults={"value": value, "updated_by": updated_by},
        )
        return obj

    @staticmethod
    def get_domain(tenant_id: UUID, domain: str, site_id: UUID | None = None) -> dict:
        """Get all settings in a domain as a dict."""
        result = {k: v for k, v in DEFAULTS.items() if k.startswith(f"{domain}.")}
        for entry in TenantConfig.objects.filter(tenant_id=tenant_id, domain=domain, site__isnull=True):
            result[entry.key] = entry.value
        if site_id:
            for entry in TenantConfig.objects.filter(tenant_id=tenant_id, domain=domain, site_id=site_id):
                result[entry.key] = entry.value
        return result

    @staticmethod
    def apply_preset(tenant_id: UUID, preset_name: str, updated_by=None):
        """Apply a preset — creates/updates TenantConfig rows."""
        preset = PRESETS.get(preset_name, {})
        for key, value in preset.items():
            ConfigService.set(tenant_id, key, value, updated_by=updated_by)
        return len(preset)

    @staticmethod
    def get_all(tenant_id: UUID, site_id: UUID | None = None) -> dict:
        """Get all settings with tenant overrides applied."""
        result = dict(DEFAULTS)
        for entry in TenantConfig.objects.filter(tenant_id=tenant_id, site__isnull=True):
            result[entry.key] = entry.value
        if site_id:
            for entry in TenantConfig.objects.filter(tenant_id=tenant_id, site_id=site_id):
                result[entry.key] = entry.value
        return result

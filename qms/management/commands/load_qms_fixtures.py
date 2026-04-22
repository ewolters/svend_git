"""Load QMS system templates, workflow presets, and signal types.

Idempotent — safe to re-run. Uses update_or_create to handle auto_now_add
fields that Django's loaddata skips with raw=True.

Also wires WorkflowPhase.available_templates M2M.

Usage:
    python manage.py load_qms_fixtures
    python manage.py load_qms_fixtures --templates-only
    python manage.py load_qms_fixtures --workflows-only
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from qms.models import ToolTemplate
from qms.workflow_models import (
    SignalTypeRegistry,
    WorkflowPhase,
    WorkflowTemplate,
    WorkflowTransition,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

# Which templates belong in which phases (by workflow key + phase key)
# Maps: phase_key -> [template_slugs]
PHASE_TEMPLATE_MAP = {
    "detect": ["process-confirmation", "fft"],
    "contain": ["process-confirmation"],
    "investigate": ["rca", "fmea", "ishikawa", "ce-matrix"],
    "standardize": ["a3-report"],
    "verify": ["process-confirmation", "fft"],
    "correct": ["a3-report", "rca"],
}


class Command(BaseCommand):
    help = "Load QMS system templates, workflow presets, signal types, and wire M2M."

    def add_arguments(self, parser):
        parser.add_argument("--templates-only", action="store_true", help="Only load tool templates")
        parser.add_argument("--workflows-only", action="store_true", help="Only load workflows + signal types")

    def handle(self, *args, **options):
        templates_only = options.get("templates_only", False)
        workflows_only = options.get("workflows_only", False)

        if not templates_only and not workflows_only:
            # Load everything
            self._load_templates()
            self._load_workflows()
            self._wire_m2m()
        elif templates_only:
            self._load_templates()
        elif workflows_only:
            self._load_workflows()
            self._wire_m2m()

    def _load_templates(self):
        fixture_path = FIXTURE_DIR / "system_templates.json"
        with open(fixture_path) as f:
            data = json.load(f)

        created_count = 0
        updated_count = 0
        for item in data:
            fields = item["fields"]
            pk = item["pk"]
            _, created = ToolTemplate.objects.update_or_create(
                id=pk,
                defaults={
                    "tenant": None,
                    "name": fields["name"],
                    "slug": fields["slug"],
                    "description": fields.get("description", ""),
                    "icon": fields.get("icon", ""),
                    "is_system": fields.get("is_system", True),
                    "version": fields.get("version", 1),
                    "schema": fields["schema"],
                    "status_flow": fields.get("status_flow", []),
                },
            )
            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(self.style.SUCCESS(f"Templates: {created_count} created, {updated_count} updated"))

    def _load_workflows(self):
        fixture_path = FIXTURE_DIR / "workflow_presets.json"
        with open(fixture_path) as f:
            data = json.load(f)

        counts = {"workflowtemplate": 0, "workflowphase": 0, "workflowtransition": 0, "signaltyperegistry": 0}

        for item in data:
            model_name = item["model"]
            pk = item["pk"]
            fields = item["fields"]

            if model_name == "qms.workflowtemplate":
                WorkflowTemplate.objects.update_or_create(
                    id=pk,
                    defaults={
                        "tenant": None,
                        "name": fields["name"],
                        "is_system": fields.get("is_system", True),
                        "is_active": fields.get("is_active", True),
                    },
                )
                counts["workflowtemplate"] += 1

            elif model_name == "qms.workflowphase":
                wf = WorkflowTemplate.objects.get(id=fields["workflow"])
                WorkflowPhase.objects.update_or_create(
                    id=pk,
                    defaults={
                        "workflow": wf,
                        "key": fields["key"],
                        "label": fields["label"],
                        "sort_order": fields["sort_order"],
                        "color": fields.get("color", ""),
                    },
                )
                counts["workflowphase"] += 1

            elif model_name == "qms.workflowtransition":
                wf = WorkflowTemplate.objects.get(id=fields["workflow"])
                from_p = WorkflowPhase.objects.get(id=fields["from_phase"])
                to_p = WorkflowPhase.objects.get(id=fields["to_phase"])
                WorkflowTransition.objects.update_or_create(
                    id=pk,
                    defaults={
                        "workflow": wf,
                        "from_phase": from_p,
                        "to_phase": to_p,
                        "label": fields["label"],
                        "gate_conditions": fields.get("gate_conditions", {}),
                    },
                )
                counts["workflowtransition"] += 1

            elif model_name == "qms.signaltyperegistry":
                auto_phase = None
                if fields.get("auto_phase"):
                    auto_phase = WorkflowPhase.objects.get(id=fields["auto_phase"])
                SignalTypeRegistry.objects.update_or_create(
                    id=pk,
                    defaults={
                        "tenant": None,
                        "key": fields["key"],
                        "label": fields["label"],
                        "default_severity": fields.get("default_severity", "warning"),
                        "is_system": fields.get("is_system", True),
                        "icon": fields.get("icon", ""),
                        "auto_phase": auto_phase,
                    },
                )
                counts["signaltyperegistry"] += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Workflows: {counts['workflowtemplate']} templates, "
                f"{counts['workflowphase']} phases, "
                f"{counts['workflowtransition']} transitions, "
                f"{counts['signaltyperegistry']} signal types"
            )
        )

    def _wire_m2m(self):
        """Wire available_templates on each WorkflowPhase based on PHASE_TEMPLATE_MAP."""
        template_cache = {t.slug: t for t in ToolTemplate.objects.filter(is_system=True)}
        wired = 0

        for phase in WorkflowPhase.objects.all():
            slugs = PHASE_TEMPLATE_MAP.get(phase.key, [])
            templates = [template_cache[s] for s in slugs if s in template_cache]
            if templates:
                phase.available_templates.set(templates)
                wired += len(templates)

        self.stdout.write(self.style.SUCCESS(f"M2M: {wired} template-phase links wired"))

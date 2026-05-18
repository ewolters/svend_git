"""QMS composable artifact tests — PCL wiring."""

from django.test import TestCase

from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from pcl import service
from pcl.models import Datapoint, Measure
from qms.models import Artifact, ArtifactSection, ToolTemplate

FMEA_SCHEMA = {
    "sections": [
        {
            "key": "failure_modes",
            "type": "grid",
            "label": "Failure Modes",
            "config": {
                "columns": [
                    {"key": "mode", "type": "text"},
                    {"key": "severity", "type": "score", "min": 1, "max": 10},
                    {"key": "occurrence", "type": "score", "min": 1, "max": 10},
                    {"key": "detection", "type": "score", "min": 1, "max": 10},
                    {"key": "rpn", "type": "computed", "formula": "severity * occurrence * detection"},
                ],
            },
        },
    ],
}

TEXT_ONLY_SCHEMA = {
    "sections": [
        {"key": "background", "type": "text", "label": "Background"},
    ],
}


@SECURE_OFF
class QMSPCLWriteTest(TestCase):
    def setUp(self):
        self.user = make_user("qms@test.com", tier="team")
        self.tenant = make_tenant("QMS Org", slug="qms-org", plan="team")
        make_membership(self.tenant, self.user)

        self.template = ToolTemplate.objects.create(
            name="Test FMEA",
            slug="test-fmea",
            schema=FMEA_SCHEMA,
            is_system=True,
        )
        self.artifact = Artifact.objects.create(
            template=self.template,
            tenant=self.tenant,
            owner=self.user,
            title="Bearing FMEA",
        )

    def test_grid_computed_writes_to_pcl(self):
        ArtifactSection.objects.create(
            artifact=self.artifact,
            section_key="failure_modes",
            primitive_type="grid",
            data={
                "rows": [
                    {"mode": "crack", "severity": 8, "occurrence": 4, "detection": 6},
                    {"mode": "wear", "severity": 5, "occurrence": 7, "detection": 3},
                ],
            },
        )
        prefix = f"qms/{str(self.artifact.id)[:8]}/failure_modes/rpn"
        # row1 RPN=192, row2 RPN=105
        avg = service.read(prefix, tenant_id=self.tenant.id)
        assert avg == 148.5
        mx = service.read(f"{prefix}_max", tenant_id=self.tenant.id)
        assert mx == 192.0

    def test_grid_no_computed_cols_skips_pcl(self):
        schema = {
            "sections": [
                {
                    "key": "items",
                    "type": "grid",
                    "label": "Items",
                    "config": {
                        "columns": [
                            {"key": "name", "type": "text"},
                            {"key": "score", "type": "score", "min": 1, "max": 5},
                        ],
                    },
                },
            ],
        }
        template = ToolTemplate.objects.create(
            name="Simple Grid",
            slug="simple-grid",
            schema=schema,
            is_system=True,
        )
        artifact = Artifact.objects.create(
            template=template,
            tenant=self.tenant,
            owner=self.user,
            title="Simple",
        )
        ArtifactSection.objects.create(
            artifact=artifact,
            section_key="items",
            primitive_type="grid",
            data={"rows": [{"name": "a", "score": 3}]},
        )
        assert Measure.objects.filter(slug__startswith="qms/").count() == 0

    def test_text_section_skips_pcl(self):
        template = ToolTemplate.objects.create(
            name="Text Only",
            slug="text-only",
            schema=TEXT_ONLY_SCHEMA,
            is_system=True,
        )
        artifact = Artifact.objects.create(
            template=template,
            tenant=self.tenant,
            owner=self.user,
            title="Notes",
        )
        ArtifactSection.objects.create(
            artifact=artifact,
            section_key="background",
            primitive_type="text",
            data={"content": "Some background text"},
        )
        assert Measure.objects.filter(slug__startswith="qms/").count() == 0

    def test_empty_rows_skips_pcl(self):
        ArtifactSection.objects.create(
            artifact=self.artifact,
            section_key="failure_modes",
            primitive_type="grid",
            data={"rows": []},
        )
        assert Measure.objects.filter(slug__startswith="qms/").count() == 0

    def test_single_row_mean_equals_max(self):
        ArtifactSection.objects.create(
            artifact=self.artifact,
            section_key="failure_modes",
            primitive_type="grid",
            data={"rows": [{"mode": "crack", "severity": 8, "occurrence": 4, "detection": 6}]},
        )
        prefix = f"qms/{str(self.artifact.id)[:8]}/failure_modes/rpn"
        avg = service.read(prefix, tenant_id=self.tenant.id)
        mx = service.read(f"{prefix}_max", tenant_id=self.tenant.id)
        assert avg == mx == 192.0

    def test_update_writes_new_datapoint(self):
        section = ArtifactSection.objects.create(
            artifact=self.artifact,
            section_key="failure_modes",
            primitive_type="grid",
            data={"rows": [{"mode": "crack", "severity": 8, "occurrence": 4, "detection": 6}]},
        )
        slug = f"qms/{str(self.artifact.id)[:8]}/failure_modes/rpn"
        m = Measure.objects.get(slug=slug, tenant_id=self.tenant.id)
        assert Datapoint.objects.filter(measure=m).count() == 1

        # Update scores — second save writes a second datapoint
        section.data = {"rows": [{"mode": "crack", "severity": 4, "occurrence": 2, "detection": 3}]}
        section.save()
        assert Datapoint.objects.filter(measure=m).count() == 2
        latest = Datapoint.objects.filter(measure=m).order_by("-timestamp").first()
        assert latest.value == 24.0

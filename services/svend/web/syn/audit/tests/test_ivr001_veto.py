"""
IVR-001: Mechanical veto in validate_for_transition().

Tests that security_analyst rejection on a RiskAssessment blocks
feature/migration CRs from transitioning past approved.

Standard: CHG-001 §7.1.1
<!-- test: syn.audit.tests.test_ivr001_veto.SecurityAnalystVetoTest -->
"""

from django.test import TestCase

from syn.audit.models import AgentVote, ChangeRequest, RiskAssessment


class SecurityAnalystVetoTest(TestCase):
    """IVR-001: security_analyst reject vote blocks feature/migration transitions."""

    def _make_cr(self, change_type="feature", **kwargs):
        defaults = {
            "title": f"Test CR for {change_type}",
            "description": "Functional test CR for IVR-001 veto testing.",
            "change_type": change_type,
            "author": "test",
            "status": "approved",
            "justification": "Testing IVR-001 mechanical veto.",
            "affected_files": ["syn/audit/models.py"],
            "implementation_plan": {"steps": ["test"]},
            "testing_plan": {"steps": ["test"]},
            "rollback_plan": {"steps": ["test"]},
        }
        defaults.update(kwargs)
        return ChangeRequest.objects.create(**defaults)

    def _add_ra_with_vote(self, cr, role, recommendation):
        ra = RiskAssessment.objects.create(
            change_request=cr,
            assessed_by="test",
            assessment_type="single_agent",
            security_score=1,
            availability_score=1,
            integrity_score=1,
            confidentiality_score=1,
            privacy_score=1,
            overall_score=1,
            overall_recommendation="approve",
            conditions=[],
        )
        AgentVote.objects.create(
            risk_assessment=ra,
            agent_role=role,
            recommendation=recommendation,
            risk_scores={
                "security": 1,
                "availability": 1,
                "integrity": 1,
                "confidentiality": 1,
                "privacy": 1,
            },
            rationale=f"Test {role} vote: {recommendation}",
        )
        return ra

    def test_feature_cr_blocked_by_security_analyst_reject(self):
        """Feature CR with security_analyst reject vote cannot move past approved."""
        cr = self._make_cr("feature")
        self._add_ra_with_vote(cr, "security_analyst", "reject")
        errors = cr.validate_for_transition("in_progress")
        self.assertTrue(
            any("veto" in e.lower() or "IVR-001" in e for e in errors),
            f"Expected veto error, got: {errors}",
        )

    def test_migration_cr_blocked_by_security_analyst_reject(self):
        """Migration CR with security_analyst reject vote is also blocked."""
        cr = self._make_cr("migration")
        self._add_ra_with_vote(cr, "security_analyst", "reject")
        errors = cr.validate_for_transition("in_progress")
        self.assertTrue(
            any("IVR-001" in e for e in errors),
            f"Expected IVR-001 veto error, got: {errors}",
        )

    def test_feature_cr_passes_with_security_analyst_approve(self):
        """Feature CR with security_analyst approve vote transitions normally."""
        cr = self._make_cr("feature")
        self._add_ra_with_vote(cr, "security_analyst", "approve")
        errors = cr.validate_for_transition("in_progress")
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_enhancement_cr_not_affected_by_veto_gate(self):
        """Enhancement (non-MULTI_AGENT_TYPES) not subject to veto check."""
        cr = self._make_cr("enhancement")
        self._add_ra_with_vote(cr, "security_analyst", "reject")
        errors = cr.validate_for_transition("in_progress")
        veto_errors = [e for e in errors if "IVR-001" in e]
        self.assertEqual(
            veto_errors, [], f"Enhancement should not trigger veto: {veto_errors}"
        )

    def test_cr_without_votes_still_passes(self):
        """CR with RA but no votes yet passes (veto requires explicit reject)."""
        cr = self._make_cr("feature")
        RiskAssessment.objects.create(
            change_request=cr,
            assessed_by="test",
            assessment_type="single_agent",
            security_score=1,
            availability_score=1,
            integrity_score=1,
            confidentiality_score=1,
            privacy_score=1,
            overall_score=1,
            overall_recommendation="approve",
            conditions=[],
        )
        errors = cr.validate_for_transition("in_progress")
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_architect_reject_does_not_trigger_veto(self):
        """Only security_analyst role triggers mechanical veto, not architect."""
        cr = self._make_cr("feature")
        self._add_ra_with_vote(cr, "architect", "reject")
        errors = cr.validate_for_transition("in_progress")
        veto_errors = [e for e in errors if "IVR-001" in e]
        self.assertEqual(
            veto_errors, [], f"Architect reject should not trigger veto: {veto_errors}"
        )

    def test_veto_blocks_testing_transition_too(self):
        """Veto blocks any APPROVED_PLUS target, including testing."""
        cr = self._make_cr("feature", status="in_progress")
        self._add_ra_with_vote(cr, "security_analyst", "reject")
        errors = cr.validate_for_transition("testing")
        self.assertTrue(
            any("IVR-001" in e for e in errors),
            f"Expected IVR-001 on testing transition, got: {errors}",
        )

    def test_veto_blocks_completed_transition(self):
        """Veto blocks completed transition too."""
        cr = self._make_cr(
            "feature",
            status="testing",
            commit_shas=["abc123"],
            log_md_ref="CR:test",
        )
        self._add_ra_with_vote(cr, "security_analyst", "reject")
        errors = cr.validate_for_transition("completed")
        self.assertTrue(
            any("IVR-001" in e for e in errors),
            f"Expected IVR-001 on completed transition, got: {errors}",
        )

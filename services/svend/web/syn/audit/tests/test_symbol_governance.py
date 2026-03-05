"""Symbol governance structural tests (TST-001 §4.1: Structural).

Verifies that all governed symbols exist, are importable, and have correct types.
Each test class covers symbols from a specific module, linked to the standard
that governs them via <!-- impl: --> and <!-- test: --> hooks.

Standards: CMP-001, AUD-001, ERR-001, SEC-001, LOG-001, SCH-001, DAT-001, QMS-001
Compliance: SOC 2 CC8.1, CC4.1
"""

import importlib
import inspect

from django.test import SimpleTestCase


def _sym(module_path, name):
    """Import and return a symbol from a dotted module path."""
    mod = importlib.import_module(module_path)
    return getattr(mod, name, None)


# ---------------------------------------------------------------------------
# CMP-001 §6: Compliance check functions (syn/audit/compliance.py)
# ---------------------------------------------------------------------------


class ComplianceCheckSymbolsTest(SimpleTestCase):
    """CMP-001 §6: All compliance check functions exist and are callable."""

    MODULE = "syn.audit.compliance"

    EXPECTED_CHECKS = [
        "register",
        "get_check_soc2_controls",
        "get_all_soc2_controls",
        "soc2_control_coverage",
        "check_audit_integrity",
        "check_security_config",
        "check_encryption_status",
        "check_permission_coverage",
        "check_access_logging",
        "check_backup_freshness",
        "check_password_policy",
        "check_data_retention",
        "check_privacy_data_export",
        "check_ssl_tls",
        "run_standards_tests_for",
        "check_session_security",
        "check_error_handling",
        "check_rate_limiting",
        "check_secret_management",
        "check_log_completeness",
        "check_security_headers",
        "check_incident_readiness",
        "check_sla_compliance",
        "run_check",
        "check_architecture",
        "check_caching",
        "check_roadmap",
        "check_output_quality",
        "check_policy_review",
    ]

    def test_all_check_functions_exist(self):
        """Every compliance check function is importable."""
        missing = []
        for name in self.EXPECTED_CHECKS:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing symbols: {missing}")

    def test_all_checks_are_registered(self):
        """All check_* functions are registered in ALL_CHECKS (keys without prefix)."""
        mod = importlib.import_module(self.MODULE)
        all_checks = getattr(mod, "ALL_CHECKS", {})
        # ALL_CHECKS keys are without the check_ prefix
        check_fns = [n for n in self.EXPECTED_CHECKS if n.startswith("check_")]
        unregistered = [n for n in check_fns if n.replace("check_", "", 1) not in all_checks]
        self.assertEqual(unregistered, [], f"Unregistered checks: {unregistered}")

    def test_check_functions_are_callable(self):
        """All check functions are callable."""
        for name in self.EXPECTED_CHECKS:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(callable(sym), f"{name} is not callable")

    def test_run_check_is_callable(self):
        """run_check dispatcher function exists and is callable."""
        sym = _sym(self.MODULE, "run_check")
        self.assertIsNotNone(sym)
        self.assertTrue(callable(sym))

    def test_soc2_control_functions_exist(self):
        """SOC 2 control mapping functions exist and are callable."""
        for name in ["get_all_soc2_controls", "get_check_soc2_controls", "soc2_control_coverage"]:
            sym = _sym(self.MODULE, name)
            self.assertIsNotNone(sym, f"{name} not found")
            self.assertTrue(callable(sym), f"{name} is not callable")


class ComplianceCheckDependencyVulnTest(SimpleTestCase):
    """CMP-001: check_dependency_vuln exists and is callable."""

    def test_check_dependency_vuln_exists(self):
        sym = _sym("syn.audit.compliance", "check_dependency_vuln")
        self.assertIsNotNone(sym)
        self.assertTrue(callable(sym))


class ComplianceCheckChangeManagementTest(SimpleTestCase):
    """CMP-001: check_change_management exists and is callable."""

    def test_check_change_management_exists(self):
        sym = _sym("syn.audit.compliance", "check_change_management")
        self.assertIsNotNone(sym)
        self.assertTrue(callable(sym))


# ---------------------------------------------------------------------------
# CMP-001 §4: Standards parser support (syn/audit/standards.py)
# ---------------------------------------------------------------------------


class StandardsParserSymbolsTest(SimpleTestCase):
    """CMP-001 §4: Standards parser support functions exist."""

    MODULE = "syn.audit.standards"

    def test_support_functions_exist(self):
        """parse_standard_titles, parse_all_sla_definitions, run_linked_test, run_standards_checks."""
        expected = [
            "parse_standard_titles",
            "parse_all_sla_definitions",
            "run_linked_test",
            "run_standards_checks",
        ]
        for name in expected:
            sym = _sym(self.MODULE, name)
            self.assertIsNotNone(sym, f"{name} not found in {self.MODULE}")
            self.assertTrue(callable(sym), f"{name} is not callable")


# ---------------------------------------------------------------------------
# AUD-001 §4: Audit event catalog (syn/audit/events.py)
# ---------------------------------------------------------------------------


class AuditEventSymbolsTest(SimpleTestCase):
    """AUD-001 §4: Audit event payload builder functions exist."""

    MODULE = "syn.audit.events"

    EXPECTED = [
        "redact_event_payload",
        "emit_audit_event",
        "build_entry_created_payload",
        "build_genesis_created_payload",
        "build_integrity_violation_payload",
        "build_chain_verified_payload",
        "build_chain_verification_failed_payload",
        "build_violation_resolved_payload",
        "build_drift_violation_payload",
        "build_drift_remediation_available_payload",
        "build_drift_sla_breached_payload",
        "build_drift_governance_escalated_payload",
        "build_drift_resolved_payload",
        "build_governance_integrity_violation_payload",
        "build_governance_drift_alert_payload",
        "build_trail_queried_payload",
        "build_trail_export_payload",
        "build_retention_policy_payload",
    ]

    def test_event_builders_exist(self):
        """All event payload builder functions are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing event builders: {missing}")

    def test_event_builders_are_callable(self):
        """All event builders are functions."""
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(callable(sym), f"{name} is not callable")

    def test_builder_naming_convention(self):
        """All builder functions follow build_*_payload naming."""
        builders = [n for n in self.EXPECTED if n.startswith("build_")]
        for name in builders:
            self.assertTrue(
                name.endswith("_payload"),
                f"{name} should end with _payload",
            )


# ---------------------------------------------------------------------------
# AUD-001 §5: Audit signal handlers (syn/audit/signals.py)
# ---------------------------------------------------------------------------


class AuditSignalSymbolsTest(SimpleTestCase):
    """AUD-001 §5: Audit signal handlers exist."""

    MODULE = "syn.audit.signals"

    EXPECTED = [
        "on_syslog_entry_created",
        "on_integrity_violation_saved",
        "on_drift_violation_saved",
        "register_audit_signals",
    ]

    def test_signal_handlers_exist(self):
        """All signal handler functions are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing signal handlers: {missing}")

    def test_register_is_callable(self):
        """register_audit_signals is callable."""
        sym = _sym(self.MODULE, "register_audit_signals")
        self.assertTrue(callable(sym))


# ---------------------------------------------------------------------------
# ERR-001 §3: Error system types (syn/err/types.py)
# ---------------------------------------------------------------------------


class ErrorTypeSymbolsTest(SimpleTestCase):
    """ERR-001 §3: Error system enums, configs, and registry types exist."""

    MODULE = "syn.err.types"

    EXPECTED = [
        "ErrorSeverity",
        "CircuitBreakerState",
        "RetryStrategy",
        "RecoveryMode",
        "SystemLayer",
        "ErrorContext",
        "RetryConfig",
        "CircuitBreakerConfig",
        "ErrorDetail",
        "ErrorRegistryEntry",
    ]

    def test_error_enums_exist(self):
        """All error type symbols are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing error types: {missing}")

    def test_enums_are_classes(self):
        """Enum-like types are classes."""
        enum_names = ["ErrorSeverity", "CircuitBreakerState", "RetryStrategy", "RecoveryMode", "SystemLayer"]
        for name in enum_names:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(inspect.isclass(sym), f"{name} should be a class")

    def test_config_types_are_classes(self):
        """Config types are classes or dataclasses."""
        config_names = ["ErrorContext", "RetryConfig", "CircuitBreakerConfig", "ErrorDetail", "ErrorRegistryEntry"]
        for name in config_names:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(
                    inspect.isclass(sym) or callable(sym),
                    f"{name} should be a class or callable",
                )


# ---------------------------------------------------------------------------
# SEC-001 §6: Permission decorators (accounts/permissions.py)
# ---------------------------------------------------------------------------


class PermissionSymbolsTest(SimpleTestCase):
    """SEC-001 §6: Permission decorator functions exist."""

    MODULE = "accounts.permissions"

    EXPECTED = [
        "require_auth",
        "require_paid",
        "require_team",
        "require_enterprise",
        "require_feature",
        "require_ml",
        "allow_guest",
        "require_org_admin",
    ]

    def test_decorators_exist(self):
        """All permission decorators are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing decorators: {missing}")

    def test_decorators_are_callable(self):
        """All permission decorators are callable (functions)."""
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(callable(sym), f"{name} is not callable")


# ---------------------------------------------------------------------------
# LOG-001 §5: Logging context accessors (syn/log/handlers.py)
# ---------------------------------------------------------------------------


class LogHandlerSymbolsTest(SimpleTestCase):
    """LOG-001 §5: Logging context variable accessors exist."""

    MODULE = "syn.log.handlers"

    EXPECTED = [
        "set_correlation_id",
        "get_correlation_id",
        "set_tenant_id",
        "get_tenant_id",
        "set_actor_id",
        "get_actor_id",
        "configure_django_logging",
    ]

    def test_context_accessors_exist(self):
        """All context variable accessors are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing log handlers: {missing}")

    def test_getters_return_values(self):
        """Getter functions return string or None."""
        for getter in ["get_correlation_id", "get_tenant_id", "get_actor_id"]:
            sym = _sym(self.MODULE, getter)
            result = sym()
            self.assertIsInstance(result, (str, type(None)))


# ---------------------------------------------------------------------------
# SCH-001 §3: Scheduler models (syn/sched/models.py)
# ---------------------------------------------------------------------------


class SchedulerModelSymbolsTest(SimpleTestCase):
    """SCH-001 §3: Scheduler model classes exist."""

    MODULE = "syn.sched.models"

    EXPECTED = [
        "CognitiveTask",
        "TaskExecution",
        "Schedule",
        "DeadLetterEntry",
        "CircuitBreakerState",
    ]

    def test_scheduler_models_exist(self):
        """All scheduler model classes are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing scheduler models: {missing}")

    def test_models_are_classes(self):
        """All scheduler models are Django model classes."""
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(inspect.isclass(sym), f"{name} should be a class")


# ---------------------------------------------------------------------------
# SCH-001 §4: Scheduler core (syn/sched/core.py)
# ---------------------------------------------------------------------------


class SchedulerCoreSymbolsTest(SimpleTestCase):
    """SCH-001 §4: Scheduler core components exist."""

    MODULE = "syn.sched.core"

    EXPECTED = [
        "TaskRegistry",
        "task",
        "CognitiveScheduler",
        "CognitiveWorker",
    ]

    def test_core_components_exist(self):
        """All scheduler core components are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing scheduler core: {missing}")

    def test_task_decorator_callable(self):
        """task() is a callable decorator."""
        sym = _sym(self.MODULE, "task")
        self.assertTrue(callable(sym))


# ---------------------------------------------------------------------------
# DAT-001 §8: Workbench models (workbench/models.py)
# ---------------------------------------------------------------------------


class WorkbenchModelSymbolsTest(SimpleTestCase):
    """DAT-001 §8: Workbench model classes exist."""

    MODULE = "workbench.models"

    EXPECTED = [
        "Project",
        "Hypothesis",
        "Evidence",
        "Conversation",
        "Workbench",
        "Artifact",
        "KnowledgeGraph",
        "EpistemicLog",
    ]

    def test_workbench_models_exist(self):
        """All workbench model classes are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing workbench models: {missing}")

    def test_models_are_classes(self):
        """All workbench models are classes."""
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(inspect.isclass(sym), f"{name} should be a class")


# ---------------------------------------------------------------------------
# QMS-001 §4: Quality management models (agents_api/models.py)
# ---------------------------------------------------------------------------


class QMSModelSymbolsTest(SimpleTestCase):
    """QMS-001 §4: Quality management model classes exist."""

    MODULE = "agents_api.models"

    EXPECTED = [
        "Workflow",
        "TriageResult",
        "SavedModel",
        "AgentLog",
        "Problem",
        "CacheEntry",
        "LLMUsage",
        "RateLimitOverride",
        "Board",
        "BoardParticipant",
        "BoardVote",
        "BoardGuestInvite",
        "SectionProgress",
        "AssessmentAttempt",
        "LearnSession",
    ]

    def test_qms_models_exist(self):
        """All QMS model classes are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing QMS models: {missing}")

    def test_models_are_classes(self):
        """All QMS models are classes."""
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is not None:
                self.assertTrue(inspect.isclass(sym), f"{name} should be a class")


# ---------------------------------------------------------------------------
# QMS-001 §5: Additional QMS models (agents_api/models.py)
# ---------------------------------------------------------------------------


class QMSExtendedModelSymbolsTest(SimpleTestCase):
    """QMS-001 §5: Extended QMS model classes exist."""

    MODULE = "agents_api.models"

    EXPECTED = [
        "PlantSimulation",
        "Site",
        "StrategicObjective",
        "AnnualObjective",
        "NCRStatusChange",
        "CAPAStatusChange",
        "AuditFinding",
        "TrainingRecordChange",
        "DocumentStatusChange",
        "SupplierStatusChange",
        "QMSFieldChange",
        "AuditChecklist",
        "ISODocument",
        "ISOSection",
    ]

    def test_extended_models_exist(self):
        """All extended QMS model classes are importable."""
        missing = []
        for name in self.EXPECTED:
            sym = _sym(self.MODULE, name)
            if sym is None:
                missing.append(name)
        self.assertEqual(missing, [], f"Missing extended QMS models: {missing}")


# ---------------------------------------------------------------------------
# QMS-001: Additional governed symbols
# ---------------------------------------------------------------------------


class QMSUtilitySymbolsTest(SimpleTestCase):
    """QMS-001: Utility functions in agents_api/models.py."""

    def test_check_rate_limit_exists(self):
        """check_rate_limit utility function exists."""
        sym = _sym("agents_api.models", "check_rate_limit")
        self.assertIsNotNone(sym)
        self.assertTrue(callable(sym))

    def test_generate_room_code_exists(self):
        """generate_room_code utility function exists."""
        sym = _sym("agents_api.models", "generate_room_code")
        self.assertIsNotNone(sym)
        self.assertTrue(callable(sym))

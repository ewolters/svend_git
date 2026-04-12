"""
Planning system management command.

Provides CLI access to the Initiative → Feature → Task hierarchy.
Primary interface for cross-session context — hand Claude a short ID,
get full context.

Usage:
  python manage.py plan list [--type init|feat|task] [--status STATUS] [--initiative INIT-xxx]
  python manage.py plan show <SHORT_ID>
  python manage.py plan tree [INIT-xxx]
  python manage.py plan search <query>
  python manage.py plan deps <FEAT-xxx>
  python manage.py plan blocked
  python manage.py plan progress
  python manage.py plan update <SHORT_ID> --status <STATUS>
  python manage.py plan import-master-plan
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from api.models import Feature, Initiative, PlanTask

# Feature table from NEXT_GEN_QMS_MASTER_PLAN.md
MASTER_PLAN_PHASES = [
    {
        "phase": 3,
        "title": "Phase 3: Enterprise Foundation",
        "quarter": "Q1-2026",
        "features": [
            (
                "E3-001",
                "Notification system (bell icon + SSE)",
                "Cross-cutting",
                "critical",
            ),
            (
                "E3-002",
                "Email notification with ActionToken response",
                "Cross-cutting",
                "critical",
            ),
            (
                "E3-003",
                "ElectronicSignature (CFR Part 11, SynaraImmutableLog)",
                "§7.5.3",
                "critical",
            ),
            (
                "E3-004",
                "CAPA as standalone model (extract from NCR)",
                "§10.2",
                "critical",
            ),
            (
                "E3-005",
                "CAPA lifecycle: NCR → containment → RCA → corrective → verify → close",
                "§10.2",
                "critical",
            ),
            (
                "E3-006",
                "CAPA → RCA module bridge (auto-populate from NCR data)",
                "§10.2",
                "high",
            ),
            (
                "E3-007",
                "Management Review template system (customizable sections)",
                "§9.3",
                "high",
            ),
            (
                "E3-008",
                "Management Review auto-populate (aggregate QMS metrics)",
                "§9.3",
                "high",
            ),
            (
                "E3-009",
                "QMSAttachment (artifact uploads on NCR/CAPA/FMEA/RCA/A3)",
                "§7.5",
                "high",
            ),
            ("E3-010", "NCR trending + Pareto analysis", "§8.7", "medium"),
            (
                "E3-011",
                "Cost of poor quality (CoPQ) tracking per NCR",
                "§9.1",
                "medium",
            ),
            (
                "E3-012",
                "Recurrence detection (flag repeat root causes across CAPAs)",
                "§10.2",
                "medium",
            ),
        ],
    },
    {
        "phase": 4,
        "title": "Phase 4: Document Control & Supplier Quality",
        "quarter": "Q2-2026",
        "features": [
            (
                "E4-001",
                "Controlled document register (version control, approval workflow)",
                "§7.5",
                "critical",
            ),
            ("E4-002", "Document change notice (DCN) lifecycle", "§7.5", "critical"),
            ("E4-003", "Document review scheduling + overdue alerts", "§7.5", "high"),
            (
                "E4-004",
                "External document management (standards, customer specs)",
                "§7.5",
                "high",
            ),
            ("E4-005", "Master list of documents and records", "§7.5", "high"),
            (
                "E4-006",
                "Supplier scorecard (quality, delivery, cost metrics)",
                "§8.4",
                "critical",
            ),
            (
                "E4-007",
                "Supplier CAPA (issue + track + response via ActionToken)",
                "§8.4",
                "critical",
            ),
            (
                "E4-008",
                "Approved supplier list (ASL) with qualification status",
                "§8.4",
                "high",
            ),
            ("E4-009", "Incoming inspection management", "§8.4", "medium"),
            ("E4-010", "Supplier audit scheduling + tracking", "§8.4", "medium"),
            (
                "E4-011",
                "Supplier portal via ActionToken (doc exchange, CAPA response)",
                "§8.4",
                "medium",
            ),
            ("E4-012", "Retention schedule management", "§7.5", "medium"),
        ],
    },
    {
        "phase": 5,
        "title": "Phase 5: Training, Audit & Design Controls",
        "quarter": "Q2-2026",
        "features": [
            ("E5-001", "Training matrix (role x competency)", "§7.2", "critical"),
            (
                "E5-002",
                "Training → Learn module integration (assessments = competency proof)",
                "§7.2",
                "critical",
            ),
            (
                "E5-003",
                "Training effectiveness tracking (link training to defect rates)",
                "§7.2",
                "high",
            ),
            (
                "E5-004",
                "CAPA → training trigger (root cause = training gap → auto-create requirement)",
                "§7.2",
                "high",
            ),
            (
                "E5-005",
                "Internal audit program management (risk-based scheduling)",
                "§9.2",
                "critical",
            ),
            (
                "E5-006",
                "Audit checklist builder (clause-based, process-based)",
                "§9.2",
                "high",
            ),
            ("E5-007", "Audit finding → CAPA bridge", "§9.2", "high"),
            ("E5-008", "Layered process audit (LPA) support", "IATF §9.2", "medium"),
            (
                "E5-009",
                "Calibration management (gage register, intervals, alerts)",
                "§7.1.5",
                "high",
            ),
            (
                "E5-010",
                "Calibration → Gage R&R link (existing SPC module)",
                "§7.1.5",
                "high",
            ),
            ("E5-011", "Out-of-calibration impact assessment", "§7.1.5", "medium"),
            (
                "E5-012",
                "Control plan management (prototype/pre-launch/production)",
                "IATF §8.5",
                "medium",
            ),
            ("E5-013", "FMEA → control plan auto-linkage", "IATF §8.5", "medium"),
            ("E5-014", "Requirements traceability matrix", "§8.3", "medium"),
        ],
    },
    {
        "phase": 6,
        "title": "Phase 6: Intelligence Layer",
        "quarter": "Q2-2026",
        "features": [
            (
                "E6-001",
                "SPC alarm → auto-NCR creation (with process data context)",
                "",
                "critical",
            ),
            (
                "E6-002",
                "AI-assisted root cause suggestion (historical pattern matching)",
                "",
                "critical",
            ),
            (
                "E6-003",
                "Cross-FMEA pattern detection (systemic failure mode identification)",
                "",
                "high",
            ),
            (
                "E6-004",
                "Predictive risk trending (which failure modes are increasing?)",
                "",
                "high",
            ),
            (
                "E6-005",
                "Natural language QMS query ('Show me overdue CAPAs for supplier X')",
                "",
                "high",
            ),
            (
                "E6-006",
                "Automated trending that triggers action (statistical significance detection)",
                "",
                "high",
            ),
            (
                "E6-007",
                "Management review auto-narrative (AI-generated executive summary)",
                "",
                "medium",
            ),
            (
                "E6-008",
                "Complaint → reportability determination assist (FDA MDR, EU Vigilance)",
                "",
                "medium",
            ),
            (
                "E6-009",
                "Change impact analysis (auto-detect downstream effects)",
                "",
                "medium",
            ),
            (
                "E6-010",
                "Audit readiness scoring ('How ready are we for audit?')",
                "",
                "medium",
            ),
            (
                "E6-011",
                "Supplier risk prediction (based on scorecard trends)",
                "",
                "low",
            ),
            (
                "E6-012",
                "Cost of quality automation (aggregate prevention/appraisal/failure costs)",
                "",
                "low",
            ),
        ],
    },
    {
        "phase": 7,
        "title": "Phase 7: Industry-Specific Extensions",
        "quarter": "Q3-2026",
        "features": [
            ("E7-001", "APQP phase-gate management", "IATF 16949", "high"),
            (
                "E7-002",
                "PPAP submission management (18 elements)",
                "IATF 16949",
                "high",
            ),
            (
                "E7-003",
                "AIAG-VDA harmonized FMEA (7-step, AP tables)",
                "IATF 16949",
                "high",
            ),
            (
                "E7-004",
                "First article inspection (AS9102 Forms 1-3)",
                "AS9100",
                "medium",
            ),
            ("E7-005", "Configuration management", "AS9100", "medium"),
            ("E7-006", "Counterfeit parts prevention workflow", "AS9100", "medium"),
            (
                "E7-007",
                "Design controls (DHF/DMR/DHR management)",
                "ISO 13485",
                "medium",
            ),
            (
                "E7-008",
                "Complaint handling with reportability determination",
                "ISO 13485/FDA",
                "medium",
            ),
            (
                "E7-009",
                "Field corrective action / recall management",
                "ISO 13485/FDA",
                "medium",
            ),
            (
                "E7-010",
                "Customer-specific requirements (CSR) database per OEM",
                "IATF 16949",
                "low",
            ),
        ],
    },
    {
        "phase": 8,
        "title": "Phase 8: DSW Statistical Calibration",
        "quarter": "Q1-2026",
        "features": [
            (
                "E8-001",
                "Calibration reference pool — 17 cases across 6 categories",
                "STAT-001 §15",
                "critical",
            ),
            (
                "E8-002",
                "Calibration runner with date-seeded daily rotation",
                "STAT-001 §15",
                "critical",
            ),
            (
                "E8-003",
                "check_statistical_calibration compliance check — Thursday rotation",
                "CMP-001 §6",
                "critical",
            ),
            (
                "E8-004",
                "Dashboard calibration section — KPI cards + per-case result table",
                "CMP-001 §6",
                "high",
            ),
            (
                "E8-005",
                "CAL enforcement type in DriftViolation model",
                "CHG-001 §8",
                "high",
            ),
            (
                "E8-006",
                "Symbol-level impl hooks on existing STAT-001 assertions",
                "STAT-001 §4-§12",
                "medium",
            ),
            (
                "E8-007",
                "Calibration test suite — 5 tests covering pool, runner, reproducibility, drift",
                "TST-001 §9",
                "high",
            ),
        ],
    },
    {
        "phase": 9,
        "title": "Phase 9: DSW Output Standardization",
        "quarter": "Q1-2026",
        "features": [
            (
                "E9-001",
                "Analysis registry — 211 entries with metadata",
                "DSW-001",
                "critical",
            ),
            (
                "E9-002",
                "Post-processing pipeline — standardize_output() in dispatch.py",
                "DSW-001",
                "critical",
            ),
            (
                "E9-003",
                "Chart standardization — apply_chart_defaults(), trace builders, SVEND_COLORS",
                "DSW-001",
                "critical",
            ),
            (
                "E9-004",
                "Frontend rendering — education (collapsible), narrative CSS, evidence badge",
                "DSW-001",
                "critical",
            ),
            (
                "E9-005",
                "Education content — hand-written for all 211 analyses",
                "DSW-001",
                "critical",
            ),
            (
                "E9-006",
                "Bayesian shadow + evidence grade rollout — stats.py (~46 analyses)",
                "STAT-001",
                "high",
            ),
            (
                "E9-007",
                "Shadow rollout — spc.py, ml.py, reliability.py, d_type.py",
                "STAT-001",
                "high",
            ),
            (
                "E9-008",
                "What-if interactivity — unified schema, Tier 1+2 (~35 analyses)",
                "DSW-001",
                "high",
            ),
            (
                "E9-009",
                "DSW-001 standard update + dsw_output_format compliance check",
                "DSW-001",
                "high",
            ),
        ],
    },
]

# Known dependencies from the master plan
MASTER_PLAN_DEPS = {
    "E3-002": ["E3-001"],  # Email notification depends on notification system
    "E3-005": ["E3-004"],  # CAPA lifecycle depends on CAPA model
    "E3-006": ["E3-004"],  # CAPA → RCA bridge depends on CAPA model
    "E3-008": ["E3-007"],  # Management Review auto-populate depends on template
    "E3-010": ["E3-004"],  # NCR trending depends on CAPA (NCR integration)
    "E3-011": ["E3-004"],  # CoPQ depends on CAPA/NCR
    "E3-012": ["E3-005"],  # Recurrence detection depends on CAPA lifecycle
    "E4-002": ["E4-001"],  # DCN depends on document register
    "E4-003": ["E4-001"],  # Review scheduling depends on document register
    "E4-005": ["E4-001"],  # Master list depends on document register
    "E4-007": ["E4-006"],  # Supplier CAPA depends on supplier scorecard
    "E4-009": ["E4-008"],  # Incoming inspection depends on ASL
    "E4-010": ["E4-008"],  # Supplier audit depends on ASL
    "E4-011": ["E4-006"],  # Supplier portal depends on scorecard
    "E4-012": ["E4-001"],  # Retention depends on document register
    "E5-002": ["E5-001"],  # Learn integration depends on training matrix
    "E5-003": ["E5-001"],  # Effectiveness tracking depends on training matrix
    "E5-004": ["E5-001", "E3-005"],  # CAPA → training depends on both
    "E5-006": ["E5-005"],  # Checklist builder depends on audit program
    "E5-007": ["E5-005", "E3-005"],  # Audit → CAPA depends on both
    "E5-008": ["E5-005"],  # LPA depends on audit program
    "E5-010": ["E5-009"],  # Gage R&R link depends on calibration
    "E5-011": ["E5-009"],  # Impact assessment depends on calibration
    "E5-013": ["E5-012"],  # FMEA linkage depends on control plan
    "E6-001": ["E3-001", "E3-004"],  # SPC → NCR depends on notifications + CAPA
    "E6-002": ["E3-005"],  # AI root cause depends on CAPA lifecycle
    "E6-003": ["E3-005"],  # Cross-FMEA depends on CAPA lifecycle
    "E6-005": ["E3-004"],  # NL query depends on CAPA model existing
    "E6-007": ["E3-008"],  # Auto-narrative depends on review auto-populate
    "E6-008": ["E3-004"],  # Complaint depends on NCR
    "E6-009": ["E4-001"],  # Change impact depends on doc control
    "E6-010": ["E5-005"],  # Audit readiness depends on audit program
    "E6-011": ["E4-006"],  # Supplier risk depends on scorecard
    "E6-012": ["E3-011"],  # CoQ depends on CoPQ
    # Phase 9: DSW Output Standardization
    "E9-002": ["E9-001"],  # Post-processor depends on registry
    "E9-003": ["E9-002"],  # Chart defaults depends on post-processor
    "E9-004": ["E9-002"],  # Frontend rendering depends on post-processor
    "E9-005": ["E9-001"],  # Education content depends on registry
    "E9-006": ["E9-001"],  # Shadow rollout depends on registry
    "E9-007": ["E9-006"],  # Other module shadows depend on stats.py rollout
    "E9-008": ["E9-004"],  # What-if depends on frontend rendering
    "E9-009": ["E9-002", "E9-005"],  # Standard update depends on pipeline + education
}


class Command(BaseCommand):
    help = "Planning system CLI — Initiative → Feature → Task hierarchy"

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest="subcommand")

        # plan list
        ls = sub.add_parser("list", help="List items")
        ls.add_argument("--type", choices=["init", "feat", "task"], default="feat")
        ls.add_argument("--status", type=str, default="")
        ls.add_argument("--initiative", type=str, default="")

        # plan show
        show = sub.add_parser("show", help="Show full context for a short ID")
        show.add_argument("short_id", type=str)

        # plan tree
        tree = sub.add_parser("tree", help="Visual hierarchy tree")
        tree.add_argument("init_id", nargs="?", type=str, default="")

        # plan search
        srch = sub.add_parser("search", help="Full-text search")
        srch.add_argument("query", type=str)

        # plan deps
        deps = sub.add_parser("deps", help="Dependency graph for a feature")
        deps.add_argument("feat_id", type=str)

        # plan blocked
        sub.add_parser("blocked", help="All blocked features")

        # plan progress
        sub.add_parser("progress", help="Initiative-level progress")

        # plan update
        upd = sub.add_parser("update", help="Update status of an item")
        upd.add_argument("short_id", type=str)
        upd.add_argument("--status", type=str, required=True)

        # plan note
        note = sub.add_parser("note", help="Add a note to an item")
        note.add_argument("short_id", type=str)
        note.add_argument("text", type=str)
        note.add_argument("--user", action="store_true", help="Prefix with $ (user note)")

        # plan activate
        act = sub.add_parser("activate", help="Set initiative(s) to active status")
        act.add_argument("init_ids", nargs="+", type=str)

        # plan context
        sub.add_parser("context", help="Dump active initiatives context (for session start)")

        # plan import-master-plan
        sub.add_parser("import-master-plan", help="Import from NEXT_GEN_QMS_MASTER_PLAN.md")

    def handle(self, *args, **options):
        cmd = options.get("subcommand")
        if not cmd:
            self.stderr.write("Usage: plan <list|show|tree|search|deps|blocked|progress|update|import-master-plan>")
            return

        handler = getattr(self, f"cmd_{cmd.replace('-', '_')}", None)
        if handler:
            handler(options)
        else:
            raise CommandError(f"Unknown subcommand: {cmd}")

    # ─── list ────────────────────────────────────────────────────────────

    def cmd_list(self, options):
        item_type = options["type"]
        status_filter = options.get("status", "")
        init_filter = options.get("initiative", "")

        if item_type == "init":
            qs = Initiative.objects.all()
            if status_filter:
                qs = qs.filter(status=status_filter)
            self.stdout.write("")
            for i in qs:
                feat_count = i.features.count()
                pct = i.progress
                self.stdout.write(f"  {i.short_id:<10} {i.title:<50} {i.status:<12} {pct:>3}%  {feat_count} features")
            self.stdout.write(f"\n  {qs.count()} initiatives")

        elif item_type == "feat":
            qs = Feature.objects.select_related("initiative").all()
            if status_filter:
                qs = qs.filter(status=status_filter)
            if init_filter:
                qs = qs.filter(initiative__short_id=init_filter)
            self.stdout.write("")
            for f in qs:
                legacy = f" ({f.legacy_id})" if f.legacy_id else ""
                self.stdout.write(
                    f"  {f.short_id:<10} {f.title:<55} {f.status:<14} {f.initiative.short_id}  P:{f.priority}{legacy}"
                )
            self.stdout.write(f"\n  {qs.count()} features")

        elif item_type == "task":
            qs = PlanTask.objects.select_related("feature", "feature__initiative").all()
            if status_filter:
                qs = qs.filter(status=status_filter)
            if init_filter:
                qs = qs.filter(feature__initiative__short_id=init_filter)
            self.stdout.write("")
            for t in qs:
                cr = f"CR:{str(t.change_request_id)[:8]}" if t.change_request_id else ""
                self.stdout.write(f"  {t.short_id:<10} {t.title:<55} {t.status:<14} {t.feature.short_id}  {cr}")
            self.stdout.write(f"\n  {qs.count()} tasks")

    # ─── show ────────────────────────────────────────────────────────────

    def cmd_show(self, options):
        sid = options["short_id"].upper()
        if sid.startswith("INIT-"):
            self._show_initiative(sid)
        elif sid.startswith("FEAT-"):
            self._show_feature(sid)
        elif sid.startswith("TASK-"):
            self._show_task(sid)
        else:
            raise CommandError(f"Unknown ID format: {sid}. Expected INIT-xxx, FEAT-xxx, or TASK-xxx")

    def _show_initiative(self, sid):
        try:
            i = Initiative.objects.get(short_id=sid)
        except Initiative.DoesNotExist:
            raise CommandError(f"Initiative not found: {sid}")

        sep = "━" * 60
        self.stdout.write(f"\n{sep}")
        self.stdout.write(f"{i.short_id}: {i.title}")
        self.stdout.write(f"{sep}\n")
        self.stdout.write(f"  UUID:        {i.id}")
        self.stdout.write(f"  Status:      {i.status}")
        self.stdout.write(f"  Quarter:     {i.target_quarter or '(none)'}")
        self.stdout.write(f"  Progress:    {i.progress}%")
        if i.description:
            self.stdout.write(f"\n  Description:\n    {i.description}")
        if i.notes:
            self.stdout.write("\n  Notes:")
            for line in i.notes.strip().split("\n"):
                self.stdout.write(f"    {line}")

        features = i.features.all()
        self.stdout.write(f"\n  Features ({features.count()}):")
        for f in features:
            marker = "✓" if f.status == "completed" else "○" if f.status == "in_progress" else "·"
            self.stdout.write(f"    {marker} {f.short_id}  {f.title:<50} {f.status}")
        self.stdout.write("")

    def _show_feature(self, sid):
        try:
            f = Feature.objects.select_related("initiative").get(short_id=sid)
        except Feature.DoesNotExist:
            raise CommandError(f"Feature not found: {sid}")

        sep = "━" * 60
        self.stdout.write(f"\n{sep}")
        self.stdout.write(f"{f.short_id}: {f.title}")
        self.stdout.write(f"{sep}\n")
        self.stdout.write(f"  UUID:        {f.id}")
        self.stdout.write(f"  Status:      {f.status}")
        self.stdout.write(f"  Priority:    {f.priority}")
        self.stdout.write(f"  Initiative:  {f.initiative.short_id} — {f.initiative.title}")
        if f.iso_clause:
            self.stdout.write(f"  ISO Clause:  {f.iso_clause}")
        if f.standards:
            self.stdout.write(f"  Standards:   {', '.join(f.standards)}")
        if f.legacy_id:
            self.stdout.write(f"  Legacy ID:   {f.legacy_id}")

        if f.description:
            self.stdout.write(f"\n  Description:\n    {f.description}")
        if f.acceptance_criteria:
            self.stdout.write(f"\n  Acceptance Criteria:\n    {f.acceptance_criteria}")

        if f.notes:
            self.stdout.write("\n  Notes:")
            for line in f.notes.strip().split("\n"):
                self.stdout.write(f"    {line}")

        # Dependencies
        deps = f.depends_on.all()
        if deps:
            self.stdout.write("\n  Dependencies:")
            for d in deps:
                status_icon = "✓" if d.status == "completed" else "⚠"
                self.stdout.write(f"    └── {d.short_id}  {d.title}  ({status_icon} {d.status})")

        # What this blocks
        blockers = f.blocks.all()
        if blockers:
            self.stdout.write("\n  Blocks:")
            for b in blockers:
                self.stdout.write(f"    ├── {b.short_id}  {b.title}")

        # Tasks
        tasks = f.tasks.all()
        if tasks:
            completed = tasks.filter(status="completed").count()
            self.stdout.write(f"\n  Tasks ({completed}/{tasks.count()} completed):")
            for t in tasks:
                if t.status == "completed":
                    marker = "✓"
                elif t.status == "in_progress":
                    marker = "○"
                else:
                    marker = "·"
                cr = f"  CR:{str(t.change_request_id)[:8]}" if t.change_request_id else ""
                self.stdout.write(f"    {marker} {t.short_id}  {t.title:<45} {t.status}{cr}")

        self.stdout.write(f"\n  Progress:    {f.progress}%")
        if f.roadmap_item_id:
            self.stdout.write(f"  Roadmap:     {f.roadmap_item_id}")
        if f.change_request_ids:
            self.stdout.write(f"  CRs:         {', '.join(str(c)[:8] for c in f.change_request_ids)}")

        self.stdout.write(f"\n  Created:     {f.created_at.strftime('%Y-%m-%d')}")
        self.stdout.write(f"  Updated:     {f.updated_at.strftime('%Y-%m-%d')}")
        self.stdout.write("")

    def _show_task(self, sid):
        try:
            t = PlanTask.objects.select_related("feature", "feature__initiative").get(short_id=sid)
        except PlanTask.DoesNotExist:
            raise CommandError(f"Task not found: {sid}")

        sep = "━" * 60
        self.stdout.write(f"\n{sep}")
        self.stdout.write(f"{t.short_id}: {t.title}")
        self.stdout.write(f"{sep}\n")
        self.stdout.write(f"  UUID:        {t.id}")
        self.stdout.write(f"  Status:      {t.status}")
        self.stdout.write(f"  Type:        {t.task_type}")
        self.stdout.write(f"  Feature:     {t.feature.short_id} — {t.feature.title}")
        self.stdout.write(f"  Initiative:  {t.feature.initiative.short_id} — {t.feature.initiative.title}")
        if t.change_request_id:
            self.stdout.write(f"  CR:          {t.change_request_id}")
        if t.description:
            self.stdout.write(f"\n  Description:\n    {t.description}")
        self.stdout.write("")

    # ─── tree ────────────────────────────────────────────────────────────

    def cmd_tree(self, options):
        init_id = options.get("init_id", "")

        if init_id:
            try:
                initiatives = [Initiative.objects.get(short_id=init_id.upper())]
            except Initiative.DoesNotExist:
                raise CommandError(f"Initiative not found: {init_id}")
        else:
            initiatives = Initiative.objects.all()

        self.stdout.write("")
        for i in initiatives:
            pct = i.progress
            self.stdout.write(f"  {i.short_id}: {i.title}  [{i.status}, {pct}%]")
            features = i.features.all()
            for idx, f in enumerate(features):
                is_last_feat = idx == len(features) - 1
                prefix = "└──" if is_last_feat else "├──"
                blocked_marker = " ⚠ blocked" if f.is_blocked else ""
                fpct = f.progress
                self.stdout.write(
                    f"  {prefix} {f.short_id}  {f.title:<50} {f.status:<14} {fpct:>3}%  {f.priority}{blocked_marker}"
                )
                tasks = f.tasks.all()
                for tidx, t in enumerate(tasks):
                    is_last_task = tidx == len(tasks) - 1
                    tprefix = "    └──" if is_last_task else "    ├──"
                    if not is_last_feat:
                        tprefix = "│   └──" if is_last_task else "│   ├──"
                    marker = "✓" if t.status == "completed" else "○" if t.status == "in_progress" else "·"
                    self.stdout.write(f"  {tprefix} {marker} {t.short_id}  {t.title:<45} {t.status}")

            # Summary
            total = features.count()
            completed = features.filter(status="completed").count()
            in_prog = features.filter(status="in_progress").count()
            planned = features.filter(status="planned").count()
            blocked = sum(1 for f in features if f.is_blocked)
            backlog = features.filter(status="backlog").count()
            self.stdout.write(
                f"\n  {total} features: {completed} completed, {in_prog} in_progress, "
                f"{planned} planned, {blocked} blocked, {backlog} backlog"
            )
            self.stdout.write("")

    # ─── search ──────────────────────────────────────────────────────────

    def cmd_search(self, options):
        query = options["query"].lower()
        self.stdout.write("")

        # Search initiatives
        from django.db.models import Q

        inits = Initiative.objects.filter(
            Q(title__icontains=query) | Q(description__icontains=query) | Q(short_id__icontains=query)
        )
        if inits:
            self.stdout.write("  Initiatives:")
            for i in inits:
                self.stdout.write(f"    {i.short_id}  {i.title}  ({i.status})")

        # Search features
        feats = Feature.objects.filter(
            Q(title__icontains=query)
            | Q(description__icontains=query)
            | Q(short_id__icontains=query)
            | Q(legacy_id__icontains=query)
            | Q(iso_clause__icontains=query)
        ).select_related("initiative")
        if feats:
            self.stdout.write("  Features:")
            for f in feats:
                legacy = f" ({f.legacy_id})" if f.legacy_id else ""
                self.stdout.write(f"    {f.short_id}  {f.title:<50} {f.status}  {f.initiative.short_id}{legacy}")

        # Search tasks
        tasks = PlanTask.objects.filter(
            Q(title__icontains=query) | Q(description__icontains=query) | Q(short_id__icontains=query)
        ).select_related("feature")
        if tasks:
            self.stdout.write("  Tasks:")
            for t in tasks:
                self.stdout.write(f"    {t.short_id}  {t.title:<50} {t.status}  {t.feature.short_id}")

        total = inits.count() + feats.count() + tasks.count()
        if total == 0:
            self.stdout.write(f"  No results for '{query}'")
        else:
            self.stdout.write(f"\n  {total} results for '{query}'")
        self.stdout.write("")

    # ─── deps ────────────────────────────────────────────────────────────

    def cmd_deps(self, options):
        sid = options["feat_id"].upper()
        try:
            f = Feature.objects.get(short_id=sid)
        except Feature.DoesNotExist:
            raise CommandError(f"Feature not found: {sid}")

        self.stdout.write(f"\n  {f.short_id}: {f.title}")

        deps = f.depends_on.all()
        if deps:
            self.stdout.write("  depends_on:")
            for d in deps:
                status_icon = "✓" if d.status == "completed" else "⚠"
                pct = d.progress
                self.stdout.write(f"    └── {d.short_id}  {d.title}  ({status_icon} {d.status} — {pct}%)")
        else:
            self.stdout.write("  depends_on: (none)")

        blockers = f.blocks.all()
        if blockers:
            self.stdout.write("  blocks:")
            for b in blockers:
                self.stdout.write(f"    └── {b.short_id}  {b.title}  ({b.status})")
        else:
            self.stdout.write("  blocks: (none)")

        if f.is_blocked:
            incomplete = f.depends_on.exclude(status="completed")
            names = ", ".join(d.short_id for d in incomplete)
            self.stdout.write(f"\n  Status: BLOCKED — waiting on {names}")
        else:
            self.stdout.write("\n  Status: READY — all dependencies satisfied")
        self.stdout.write("")

    # ─── blocked ─────────────────────────────────────────────────────────

    def cmd_blocked(self, options):
        self.stdout.write("")
        blocked_count = 0
        for f in Feature.objects.exclude(status__in=["completed", "cancelled", "deferred"]):
            if f.is_blocked:
                incomplete = f.depends_on.exclude(status="completed")
                dep_info = ", ".join(f"{d.short_id} ({d.progress}%)" for d in incomplete)
                self.stdout.write(f"  {f.short_id:<10} {f.title:<50} blocked by: {dep_info}")
                blocked_count += 1

        if blocked_count == 0:
            self.stdout.write("  No blocked features")
        else:
            self.stdout.write(f"\n  {blocked_count} features blocked")
        self.stdout.write("")

    # ─── progress ────────────────────────────────────────────────────────

    def cmd_progress(self, options):
        self.stdout.write("")
        total_features = 0
        total_completed = 0

        for i in Initiative.objects.all():
            feat_count = i.features.count()
            completed = i.features.filter(status="completed").count()
            total_features += feat_count
            total_completed += completed
            pct = i.progress

            # Progress bar (16 chars wide)
            filled = round(pct / 100 * 16)
            bar = "█" * filled + "░" * (16 - filled)

            self.stdout.write(
                f"  {i.short_id}  {i.title:<45} {i.status:<10} {pct:>3}%  [{bar}]  {completed}/{feat_count} features"
            )

        if total_features > 0:
            overall = round(total_completed / total_features * 100)
            self.stdout.write(f"\n  Overall: {overall}% ({total_completed}/{total_features} features completed)")
        else:
            self.stdout.write("\n  No features imported yet. Run: plan import-master-plan")
        self.stdout.write("")

    # ─── update ──────────────────────────────────────────────────────────

    def cmd_update(self, options):
        sid = options["short_id"].upper()
        new_status = options["status"]

        if sid.startswith("INIT-"):
            try:
                obj = Initiative.objects.get(short_id=sid)
            except Initiative.DoesNotExist:
                raise CommandError(f"Not found: {sid}")
            valid = [c[0] for c in Initiative.Status.choices]
        elif sid.startswith("FEAT-"):
            try:
                obj = Feature.objects.get(short_id=sid)
            except Feature.DoesNotExist:
                raise CommandError(f"Not found: {sid}")
            valid = [c[0] for c in Feature.Status.choices]
        elif sid.startswith("TASK-"):
            try:
                obj = PlanTask.objects.get(short_id=sid)
            except PlanTask.DoesNotExist:
                raise CommandError(f"Not found: {sid}")
            valid = [c[0] for c in PlanTask.Status.choices]
        else:
            raise CommandError(f"Unknown ID format: {sid}")

        if new_status not in valid:
            raise CommandError(f"Invalid status '{new_status}'. Valid: {', '.join(valid)}")

        old_status = obj.status
        obj.status = new_status

        # Auto-set timestamps
        now = timezone.now()
        if new_status == "completed" and hasattr(obj, "completed_at"):
            obj.completed_at = now
        if new_status == "in_progress" and hasattr(obj, "started_at") and not obj.started_at:
            obj.started_at = now

        obj.save()
        self.stdout.write(f"  {sid}: {old_status} → {new_status}")

    # ─── note ────────────────────────────────────────────────────────────

    def cmd_note(self, options):
        sid = options["short_id"].upper()
        text = options["text"]
        is_user = options.get("user", False)

        if sid.startswith("INIT-"):
            try:
                obj = Initiative.objects.get(short_id=sid)
            except Initiative.DoesNotExist:
                raise CommandError(f"Not found: {sid}")
        elif sid.startswith("FEAT-"):
            try:
                obj = Feature.objects.get(short_id=sid)
            except Feature.DoesNotExist:
                raise CommandError(f"Not found: {sid}")
        else:
            raise CommandError("Notes supported on INIT-xxx and FEAT-xxx")

        prefix = "$ " if is_user else ""
        timestamp = timezone.now().strftime("%Y-%m-%d")
        entry = f"[{timestamp}] {prefix}{text}"

        if obj.notes:
            obj.notes = obj.notes.rstrip() + "\n" + entry
        else:
            obj.notes = entry

        obj.save()
        self.stdout.write(f"  Note added to {sid}: {entry}")

    # ─── activate ────────────────────────────────────────────────────────

    def cmd_activate(self, options):
        init_ids = [i.upper() for i in options["init_ids"]]
        activated = []
        for sid in init_ids:
            try:
                init = Initiative.objects.get(short_id=sid)
            except Initiative.DoesNotExist:
                self.stderr.write(f"  Not found: {sid}")
                continue
            if init.status != "active":
                old = init.status
                init.status = "active"
                init.save()
                self.stdout.write(f"  {sid}: {old} → active")
            else:
                self.stdout.write(f"  {sid}: already active")
            activated.append(sid)

        # Show active initiatives
        active = Initiative.objects.filter(status="active")
        self.stdout.write(f"\n  Active initiatives ({active.count()}):")
        for i in active:
            feat_count = i.features.count()
            self.stdout.write(f"    {i.short_id}  {i.title}  ({feat_count} features, {i.progress}%)")
        self.stdout.write("")

    # ─── context ─────────────────────────────────────────────────────────

    def cmd_context(self, options):
        """Dump active initiatives with features — for session start context."""
        active = Initiative.objects.filter(status="active")
        if not active:
            self.stdout.write("\n  No active initiatives. Run: plan activate INIT-003")
            self.stdout.write("  Available:")
            for i in Initiative.objects.all():
                self.stdout.write(f"    {i.short_id}  {i.title}  ({i.status})")
            self.stdout.write("")
            return

        self.stdout.write("\n  ═══ ACTIVE PLANNING CONTEXT ═══\n")
        for i in active:
            self.stdout.write(f"  {i.short_id}: {i.title}  [{i.status}, {i.progress}%]")
            if i.target_quarter:
                self.stdout.write(f"  Target: {i.target_quarter}")
            if i.notes:
                self.stdout.write("  Notes:")
                for line in i.notes.strip().split("\n"):
                    self.stdout.write(f"    {line}")

            features = i.features.all()
            # Show unblocked features first (actionable)
            unblocked = [f for f in features if not f.is_blocked and f.status not in ("completed", "cancelled")]
            blocked = [f for f in features if f.is_blocked and f.status not in ("completed", "cancelled")]
            completed = features.filter(status="completed")

            if unblocked:
                self.stdout.write(f"\n  Ready to work ({len(unblocked)}):")
                for f in unblocked:
                    marker = "○" if f.status == "in_progress" else "·"
                    self.stdout.write(f"    {marker} {f.short_id}  {f.title:<50} {f.status}  P:{f.priority}")

            if blocked:
                self.stdout.write(f"\n  Blocked ({len(blocked)}):")
                for f in blocked:
                    deps = ", ".join(d.short_id for d in f.depends_on.exclude(status="completed"))
                    self.stdout.write(f"    ⚠ {f.short_id}  {f.title:<50} waiting: {deps}")

            if completed:
                self.stdout.write(f"\n  Completed ({completed.count()}):")
                for f in completed:
                    self.stdout.write(f"    ✓ {f.short_id}  {f.title}")

            self.stdout.write("")

        # Overall stats
        total_active_feats = Feature.objects.filter(initiative__status="active").count()
        total_completed = Feature.objects.filter(initiative__status="active", status="completed").count()
        self.stdout.write(
            f"  Overall: {total_completed}/{total_active_feats} features completed across active initiatives\n"
        )

    # ─── import-master-plan ──────────────────────────────────────────────

    def cmd_import_master_plan(self, options):
        self.stdout.write("\n  Importing from NEXT_GEN_QMS_MASTER_PLAN.md...\n")

        created_inits = 0
        created_feats = 0
        skipped = 0
        legacy_to_feature = {}  # Map E3-001 → Feature object for dependency wiring

        for phase_data in MASTER_PLAN_PHASES:
            phase_num = phase_data["phase"]
            init_short_id = f"INIT-{phase_num:03d}"

            # Create or get initiative
            init, init_created = Initiative.objects.get_or_create(
                short_id=init_short_id,
                defaults={
                    "title": phase_data["title"],
                    "target_quarter": phase_data["quarter"],
                    "sort_order": phase_num,
                    "status": "planned",
                },
            )
            if init_created:
                created_inits += 1
                self.stdout.write(f"  + {init.short_id}  {init.title}")
            else:
                self.stdout.write(f"  = {init.short_id}  {init.title} (exists)")

            for legacy_id, title, iso_clause, priority in phase_data["features"]:
                existing = Feature.objects.filter(legacy_id=legacy_id).first()
                if existing:
                    legacy_to_feature[legacy_id] = existing
                    skipped += 1
                    continue

                feat = Feature(
                    initiative=init,
                    title=title,
                    iso_clause=iso_clause,
                    priority=priority,
                    status="planned",
                    legacy_id=legacy_id,
                )
                feat.save()  # auto-generates short_id
                legacy_to_feature[legacy_id] = feat
                created_feats += 1
                self.stdout.write(f"    + {feat.short_id}  {title}  (was {legacy_id})")

        # Wire dependencies
        dep_count = 0
        for dependent_id, dep_list in MASTER_PLAN_DEPS.items():
            dependent = legacy_to_feature.get(dependent_id)
            if not dependent:
                continue
            for dep_id in dep_list:
                dep_feat = legacy_to_feature.get(dep_id)
                if dep_feat and not dependent.depends_on.filter(pk=dep_feat.pk).exists():
                    dependent.depends_on.add(dep_feat)
                    dep_count += 1

        self.stdout.write(
            f"\n  Done: {created_inits} initiatives, {created_feats} features created, "
            f"{skipped} skipped, {dep_count} dependencies wired\n"
        )

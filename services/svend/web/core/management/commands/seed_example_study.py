"""Seed an example study for testing the Projects/Synara workflow.

Creates a realistic manufacturing defect investigation with:
- A fully populated project charter (5W2H, impacts, goals, scope, team)
- 3 hypotheses with structured If/Then/Because fields and priors
- 5 evidence items (statistical analyses, observations, experiments)
- Evidence links with likelihood ratios and Bayesian updates applied
- End state: H1 confirmed (~0.92), H2 rejected (~0.08), H3 active (~0.65)

Usage:
    python manage.py seed_example_study
    python manage.py seed_example_study --user eric
    python manage.py seed_example_study --clean   # Remove previous example study first
"""

from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models.hypothesis import Evidence, EvidenceLink, Hypothesis
from core.models.project import Project

User = get_user_model()

EXAMPLE_TITLE = "Injection Mold Flash Defect Investigation — Line 4"


class Command(BaseCommand):
    help = "Seed an example study with hypotheses, evidence, and Bayesian updates"

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            type=str,
            default="eric",
            help="Username to own the study (default: eric)",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Remove any existing example study first",
        )

    def handle(self, *args, **options):
        username = options["user"]
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"User '{username}' not found."))
            return

        # Clean up previous example if requested
        if options["clean"]:
            deleted, _ = Project.objects.filter(title=EXAMPLE_TITLE, user=user).delete()
            if deleted:
                self.stdout.write(self.style.WARNING(f"Deleted {deleted} previous example object(s)."))

        # Check for existing
        if Project.objects.filter(title=EXAMPLE_TITLE, user=user).exists():
            self.stdout.write(self.style.WARNING("Example study already exists. Use --clean to recreate."))
            return

        # =====================================================================
        # 1. Create Project Charter
        # =====================================================================
        project = Project.objects.create(
            user=user,
            title=EXAMPLE_TITLE,
            status=Project.Status.ACTIVE,
            methodology=Project.Methodology.DMAIC,
            current_phase=Project.Phase.ANALYZE,
            phase_history=[
                {
                    "phase": "define",
                    "entered_at": (timezone.now() - timedelta(days=14)).isoformat(),
                    "notes": "Problem identified from customer complaints",
                },
                {
                    "phase": "measure",
                    "entered_at": (timezone.now() - timedelta(days=10)).isoformat(),
                    "notes": "Data collection plan executed, MSA passed",
                },
                {
                    "phase": "analyze",
                    "entered_at": (timezone.now() - timedelta(days=5)).isoformat(),
                    "notes": "Hypotheses formed, statistical analysis underway",
                },
            ],
            # Problem Definition (5W2H)
            problem_whats=[
                "Flash defects on injection-molded housing assemblies",
                "Excess material extruding at parting line seam",
            ],
            problem_wheres=[
                "Line 4, Station 3 — Engel 500T press",
                "Part number HX-4200 (rear housing)",
            ],
            problem_whens=[
                "Started 3 weeks ago (shift B only initially, now both shifts)",
                "Worse during first 2 hours after shift changeover",
            ],
            problem_magnitude=(
                "Defect rate increased from 1.2% to 4.8% (4x increase). "
                "17 customer returns in past 2 weeks. "
                "~240 affected parts/week at current production rate."
            ),
            problem_trend=Project.Trend.INCREASING,
            problem_since="2026-01-27 (first customer complaint logged)",
            problem_statement=(
                "Flash defects on part HX-4200 (rear housing) produced on Line 4, "
                "Station 3 have increased from 1.2% to 4.8% over the past 3 weeks. "
                "The defect rate is trending upward, with 17 customer returns and "
                "~240 affected parts per week. The problem was first observed on "
                "Shift B but has since spread to both shifts."
            ),
            # Business Impact
            impact_financial=(
                "Scrap cost: $18/part x 240 parts/week = $4,320/week. "
                "Customer return processing: ~$2,100/week. "
                "Projected 8-week impact if unresolved: $51,360."
            ),
            impact_customer=(
                "17 returns in 2 weeks from 3 key accounts (Apex Medical, "
                "NordicTech, Precision Assembly). Risk of losing Apex Medical "
                "contract ($1.2M annual revenue) if not resolved by Q1 review."
            ),
            impact_quality=(
                "Cpk on flash dimension dropped from 1.45 to 0.72. "
                "Process is not capable. SPC charts showing out-of-control signals."
            ),
            impact_delivery=(
                "Sorting and rework adding 0.5 days to lead time. 2 late shipments last week due to rework queue."
            ),
            # Goal Statement
            goal_statement=(
                "Reduce flash defect rate on part HX-4200 from 4.8% to below 1.0% "
                "(Cpk > 1.33) within 4 weeks, and sustain for 30 days."
            ),
            goal_metric="Flash Defect Rate",
            goal_baseline="4.8%",
            goal_target="< 1.0%",
            goal_unit="%",
            goal_deadline=date.today() + timedelta(days=21),
            # Scope
            scope_in=[
                "Line 4, Station 3 (Engel 500T press)",
                "Part HX-4200 rear housing",
                "Mold #M-4200-A (32 cavities)",
                "Material lot traceability for past 4 weeks",
            ],
            scope_out=[
                "Other press lines (Lines 1-3, 5-6)",
                "Other part numbers on same press",
                "Upstream supplier qualification",
            ],
            constraints=[
                "Cannot take Line 4 offline for more than 8 hours (production demand)",
                "No capital expenditure > $5,000 without VP approval",
                "Must maintain current cycle time (42s target)",
            ],
            assumptions=[
                "Measurement system is adequate (Gage R&R passed at 8.2% last month)",
                "Material spec has not changed from supplier",
                "Mold maintenance records are accurate",
            ],
            # Team
            champion_name="Sarah Chen",
            champion_title="VP of Manufacturing",
            leader_name="Marcus Rodriguez",
            leader_title="Senior Quality Engineer",
            team_members=[
                {"name": "Yuki Tanaka", "role": "Process Engineer", "department": "Molding"},
                {"name": "David Kim", "role": "Tooling Specialist", "department": "Maintenance"},
                {"name": "Priya Sharma", "role": "Data Analyst", "department": "Quality"},
                {"name": "James O'Brien", "role": "Shift B Lead", "department": "Production"},
            ],
            # Timeline
            target_completion=date.today() + timedelta(days=21),
            milestones=[
                {
                    "name": "MSA validation complete",
                    "target_date": (date.today() - timedelta(days=8)).isoformat(),
                    "actual_date": (date.today() - timedelta(days=9)).isoformat(),
                    "status": "completed",
                },
                {
                    "name": "Root cause confirmed",
                    "target_date": (date.today() + timedelta(days=3)).isoformat(),
                    "actual_date": None,
                    "status": "in_progress",
                },
                {
                    "name": "Corrective action implemented",
                    "target_date": (date.today() + timedelta(days=10)).isoformat(),
                    "actual_date": None,
                    "status": "planned",
                },
                {
                    "name": "30-day sustain verification",
                    "target_date": (date.today() + timedelta(days=40)).isoformat(),
                    "actual_date": None,
                    "status": "planned",
                },
            ],
            domain="manufacturing",
            can_experiment=True,
            tags=["injection-molding", "flash-defect", "DMAIC", "SPC", "customer-return"],
        )

        self.stdout.write(self.style.SUCCESS(f"Created project: {project.title}"))

        # =====================================================================
        # 2. Create Hypotheses
        # =====================================================================

        h1 = Hypothesis.objects.create(
            project=project,
            statement=(
                "If clamp tonnage has degraded below spec on the Engel 500T press, "
                "then flash will occur at the parting line because insufficient "
                "clamping force allows melt pressure to push the mold halves apart."
            ),
            if_clause="clamp tonnage has degraded below spec on the Engel 500T press",
            then_clause="flash will occur at the parting line",
            because_clause=(
                "insufficient clamping force allows melt pressure to push "
                "the mold halves apart during injection and packing phases"
            ),
            independent_variable="Clamp Tonnage",
            independent_var_values=["450T (current measured)", "500T (nominal spec)"],
            dependent_variable="Flash Defect Rate",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.DECREASE,
            predicted_magnitude="Expect >75% reduction in flash rate if tonnage restored to spec",
            rationale=(
                "Clamp force is the primary defense against flash. The Engel 500T "
                "is 8 years old. Hydraulic seals degrade over time. Shift B reported "
                "'sluggish' clamp action 4 weeks ago but no maintenance was scheduled."
            ),
            test_method=(
                "1. Measure actual clamp tonnage with strain gauge transducer. "
                "2. Compare to machine spec (500T ± 2%). "
                "3. If degraded, restore and run confirmation trial."
            ),
            success_criteria=(
                "Confirm: Measured tonnage is >5% below spec AND restoring tonnage "
                "reduces flash rate below 1%. Refute: Tonnage is within spec."
            ),
            prior_probability=0.50,
            current_probability=0.50,
            confirmation_threshold=0.90,
            rejection_threshold=0.10,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )

        h2 = Hypothesis.objects.create(
            project=project,
            statement=(
                "If the resin material lot has excessive moisture content, then flash "
                "will increase because water vaporization creates gas pressure that "
                "forces material past the parting line."
            ),
            if_clause="the resin material lot has excessive moisture content",
            then_clause="flash will increase",
            because_clause=(
                "water vaporization during injection creates gas pressure that forces material past the parting line"
            ),
            independent_variable="Resin Moisture Content",
            independent_var_values=["0.02% (spec max)", "current lot TBD"],
            dependent_variable="Flash Defect Rate",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.DECREASE,
            predicted_magnitude="Complete elimination if moisture is the sole cause",
            rationale=(
                "New resin lot received 4 weeks ago (lot #R2026-0127). "
                "Dryer maintenance was overdue. Moisture causes splay marks "
                "AND can contribute to flash through gas pressure."
            ),
            test_method=(
                "1. Test current resin lot moisture with Karl Fischer titration. "
                "2. Check dryer dew point logs for past 4 weeks. "
                "3. If high, run trial with properly dried material."
            ),
            success_criteria=(
                "Confirm: Moisture >0.02% AND flash rate drops with dried material. Refute: Moisture within spec."
            ),
            prior_probability=0.35,
            current_probability=0.35,
            confirmation_threshold=0.90,
            rejection_threshold=0.10,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )

        h3 = Hypothesis.objects.create(
            project=project,
            statement=(
                "If mold cavity wear at the parting line has exceeded tolerance, "
                "then flash will progressively worsen because the seal surface "
                "can no longer contain material under normal injection pressure."
            ),
            if_clause="mold cavity wear at the parting line has exceeded tolerance",
            then_clause="flash will progressively worsen",
            because_clause=(
                "the parting line seal surface can no longer contain material under normal injection pressure"
            ),
            independent_variable="Parting Line Wear (gap)",
            independent_var_values=["0.00mm (new)", "0.05mm (max spec)", "TBD (current)"],
            dependent_variable="Flash Defect Rate",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.INCREASE,
            predicted_magnitude="Gradual worsening correlated with mold shot count",
            rationale=(
                "Mold #M-4200-A has 1.2M shots. Last parting line inspection was "
                "6 months ago. Wear is cumulative and irreversible without rework. "
                "Could explain why problem is getting worse over time."
            ),
            test_method=(
                "1. Measure parting line gap with feeler gauges at 8 points. "
                "2. Compare to mold qualification report. "
                "3. Correlate with shot count progression."
            ),
            success_criteria=(
                "Confirm: Gap >0.05mm at multiple points AND correlates with defect timeline. "
                "Refute: Gap within spec everywhere."
            ),
            prior_probability=0.30,
            current_probability=0.30,
            confirmation_threshold=0.90,
            rejection_threshold=0.10,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )

        self.stdout.write(self.style.SUCCESS("Created 3 hypotheses"))

        # =====================================================================
        # 3. Create Evidence Items
        # =====================================================================

        e1 = Evidence.objects.create(
            project=project,
            summary=(
                "Clamp tonnage measured at 452T — 9.6% below 500T spec. "
                "Hydraulic pressure log shows gradual decline over 6 weeks."
            ),
            details=(
                "Strain gauge transducer measurement on Engel 500T (Line 4, Station 3). "
                "Three measurements taken: 451T, 453T, 452T (mean 452T). "
                "Machine spec: 500T ± 2% (490-510T). Current reading is 48T below nominal. "
                "Hydraulic pressure trending chart shows linear decline from 498T to 452T "
                "over 6-week period, consistent with seal degradation."
            ),
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="Clamp force measurement — strain gauge transducer",
            result_type=Evidence.ResultType.QUANTITATIVE,
            confidence=0.95,
            measured_value=452.0,
            expected_value=500.0,
            unit="metric tons",
            sample_size=3,
            created_by=user,
        )

        e2 = Evidence.objects.create(
            project=project,
            summary=(
                "Resin moisture test: 0.008% — well within spec (<0.02%). Dryer dew point logs normal for past 4 weeks."
            ),
            details=(
                "Karl Fischer titration on current resin lot #R2026-0127. "
                "Three samples tested: 0.007%, 0.009%, 0.008% (mean 0.008%). "
                "Spec limit: <0.02%. Result is less than half the allowable maximum. "
                "Dryer dew point logs reviewed for Jan 27 - Feb 14: all readings "
                "between -35°F and -40°F (spec: below -20°F). No anomalies."
            ),
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="Karl Fischer titration + dryer log review",
            result_type=Evidence.ResultType.QUANTITATIVE,
            confidence=0.92,
            measured_value=0.008,
            expected_value=0.02,
            unit="% moisture",
            sample_size=3,
            created_by=user,
        )

        e3 = Evidence.objects.create(
            project=project,
            summary=(
                "Shift B operator reported 'sluggish clamp movement' 4 weeks ago. "
                "Maintenance work order was created but deprioritized."
            ),
            details=(
                "Shift B lead James O'Brien reported abnormal clamp behavior on "
                "Jan 24 (work order #WO-2026-0187). Described as 'clamp closes slower "
                "than normal, hesitates before locking.' Maintenance assessed as 'monitor' "
                "priority. No corrective action taken. Timing aligns with onset of "
                "flash defect increase (first complaint Jan 27)."
            ),
            source_type=Evidence.SourceType.OBSERVATION,
            source_description="Operator report + maintenance work order review",
            result_type=Evidence.ResultType.QUALITATIVE,
            confidence=0.70,
            created_by=user,
        )

        e4 = Evidence.objects.create(
            project=project,
            summary=(
                "DOE trial: Increasing pack pressure by 10% at current (degraded) "
                "clamp tonnage significantly increases flash rate (p=0.003). "
                "Clamp force is the limiting factor."
            ),
            details=(
                "2-factor DOE (pack pressure x hold time) at current clamp condition. "
                "Pack pressure: 80 MPa vs 88 MPa. Hold time: 8s vs 10s. "
                "Result: Pack pressure main effect highly significant (p=0.003, "
                "effect size d=1.8). Hold time: not significant (p=0.42). "
                "Interaction: not significant (p=0.61). "
                "This confirms the mold cavity is not fully sealed — higher injection "
                "force pushes more material past the parting line. Consistent with "
                "insufficient clamp force."
            ),
            source_type=Evidence.SourceType.EXPERIMENT,
            source_description="2-factor DOE on pack pressure and hold time",
            result_type=Evidence.ResultType.STATISTICAL,
            confidence=0.90,
            p_value=0.003,
            effect_size=1.8,
            sample_size=32,
            statistical_test="Full factorial ANOVA (2^2, 8 replicates)",
            created_by=user,
        )

        e5 = Evidence.objects.create(
            project=project,
            summary=(
                "Parting line gap measurement: 0.03mm average across 8 points. "
                "Within spec (<0.05mm) but approaching wear limit."
            ),
            details=(
                "Feeler gauge measurement at 8 points around mold parting line. "
                "Results: 0.02, 0.03, 0.04, 0.03, 0.02, 0.03, 0.03, 0.04 mm. "
                "Mean: 0.03mm, Max: 0.04mm, Spec: <0.05mm. "
                "Mold qualification report (new): 0.01mm average. "
                "Wear is progressing but has not yet exceeded tolerance. "
                "At current wear rate, will exceed spec in ~200K shots (~3 months)."
            ),
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="Parting line gap measurement — feeler gauges, 8 points",
            result_type=Evidence.ResultType.QUANTITATIVE,
            confidence=0.85,
            measured_value=0.03,
            expected_value=0.05,
            unit="mm",
            sample_size=8,
            created_by=user,
        )

        self.stdout.write(self.style.SUCCESS("Created 5 evidence items"))

        # =====================================================================
        # 4. Link Evidence to Hypotheses and Apply Bayesian Updates
        # =====================================================================

        # --- Evidence 1: Clamp tonnage 9.6% below spec ---
        # Strongly supports H1 (clamp issue), weakly opposes H2 (moisture), mildly supports H3
        link1a = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e1,
            likelihood_ratio=8.0,  # Very strong support
            reasoning=(
                "Clamp tonnage is 9.6% below spec — almost 5x the allowable tolerance. "
                "This is a direct measurement confirming the hypothesized cause. "
                "LR=8: P(measure 452T | clamp degraded) >> P(measure 452T | clamp OK)."
            ),
        )
        h1.apply_evidence(link1a)

        link1b = EvidenceLink.objects.create(
            hypothesis=h2,
            evidence=e1,
            likelihood_ratio=0.9,  # Slightly opposes (clamp issue explains flash, not moisture)
            reasoning=(
                "Finding a mechanical cause (clamp) slightly reduces probability of "
                "moisture being the primary cause. However, multiple causes are possible."
            ),
        )
        h2.apply_evidence(link1b)

        link1c = EvidenceLink.objects.create(
            hypothesis=h3,
            evidence=e1,
            likelihood_ratio=1.1,  # Very slight support (low clamp + wear = worse)
            reasoning=(
                "Degraded clamp could be masking or compounding mold wear. "
                "Slight support — if clamp is already low, even minor wear matters more."
            ),
        )
        h3.apply_evidence(link1c)

        # --- Evidence 2: Resin moisture within spec ---
        # Strongly opposes H2, neutral to H1 and H3
        link2a = EvidenceLink.objects.create(
            hypothesis=h2,
            evidence=e2,
            likelihood_ratio=0.15,  # Strong opposition
            reasoning=(
                "Moisture is 0.008% — less than half the 0.02% spec limit. "
                "LR=0.15: P(measure 0.008% | moisture is cause) << P(0.008% | moisture not cause). "
                "This effectively rules out moisture as a contributing factor."
            ),
        )
        h2.apply_evidence(link2a)

        link2b = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e2,
            likelihood_ratio=1.05,  # Nearly neutral, very slight support
            reasoning=(
                "Eliminating moisture as a cause slightly increases the relative "
                "probability of clamp being the primary cause."
            ),
        )
        h1.apply_evidence(link2b)

        # --- Evidence 3: Operator report of sluggish clamp ---
        # Supports H1 (timeline match), neutral to others
        link3a = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e3,
            likelihood_ratio=3.5,  # Strong support
            reasoning=(
                "Operator reported sluggish clamp 3 days before first customer complaint. "
                "Timeline alignment is compelling. LR=3.5: Independent observation "
                "corroborating the clamp degradation hypothesis."
            ),
        )
        h1.apply_evidence(link3a)

        # --- Evidence 4: DOE confirms pack pressure sensitivity ---
        # Strongly supports H1 (mold not sealed), opposes H2 (not moisture-related)
        link4a = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e4,
            likelihood_ratio=5.0,  # Very strong support
            reasoning=(
                "DOE shows flash is highly sensitive to injection force (p=0.003, d=1.8). "
                "This is exactly what we'd expect if clamping force is insufficient — "
                "higher injection pressure overcomes the weakened clamp. "
                "LR=5: P(this DOE result | clamp issue) >> P(this result | no clamp issue)."
            ),
        )
        h1.apply_evidence(link4a)

        link4b = EvidenceLink.objects.create(
            hypothesis=h2,
            evidence=e4,
            likelihood_ratio=0.5,  # Moderate opposition
            reasoning=(
                "If moisture were the cause, we'd expect splay marks and gas-driven "
                "defects, not a clean pressure-dependent flash pattern. DOE result "
                "is inconsistent with moisture as root cause."
            ),
        )
        h2.apply_evidence(link4b)

        # --- Evidence 5: Parting line gap within spec ---
        # Opposes H3 (wear not at failure point), slight support H1
        link5a = EvidenceLink.objects.create(
            hypothesis=h3,
            evidence=e5,
            likelihood_ratio=0.4,  # Moderate opposition
            reasoning=(
                "Parting line gap is 0.03mm — within the 0.05mm spec. Wear is present "
                "but has not reached the failure threshold. This weakens the hypothesis "
                "that mold wear is the primary current cause (though it confirms future risk)."
            ),
        )
        h3.apply_evidence(link5a)

        link5b = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e5,
            likelihood_ratio=1.3,  # Mild support
            reasoning=(
                "Mold being within spec further isolates the clamp as the primary cause. "
                "If mold wear were already critical, we'd expect >0.05mm gaps."
            ),
        )
        h1.apply_evidence(link5b)

        # Reload to get final probabilities
        h1.refresh_from_db()
        h2.refresh_from_db()
        h3.refresh_from_db()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=== Example Study Created ==="))
        self.stdout.write(f"  Project: {project.title}")
        self.stdout.write(f"  Phase:   {project.get_current_phase_display()}")
        self.stdout.write(f"  ID:      {project.id}")
        self.stdout.write("")
        self.stdout.write("  Hypotheses:")
        self.stdout.write(
            f"    H1: Clamp tonnage degraded    → P = {h1.current_probability:.2%}  "
            f"[{h1.get_status_display()}]  ({h1.evidence_count} evidence)"
        )
        self.stdout.write(
            f"    H2: Resin moisture issue       → P = {h2.current_probability:.2%}  "
            f"[{h2.get_status_display()}]  ({h2.evidence_count} evidence)"
        )
        self.stdout.write(
            f"    H3: Mold parting line wear     → P = {h3.current_probability:.2%}  "
            f"[{h3.get_status_display()}]  ({h3.evidence_count} evidence)"
        )
        self.stdout.write("")
        self.stdout.write(f"  Evidence items: {Evidence.objects.filter(project=project).count()}")
        self.stdout.write(f"  Evidence links: {EvidenceLink.objects.filter(hypothesis__project=project).count()}")
        self.stdout.write("")
        self.stdout.write(self.style.NOTICE("  View at: /app/projects/"))

"""Seed enterprise demo account for Next Level Partners.

Creates a realistic packaging manufacturing scenario ("Apex Manufacturing")
with fully populated Hoshin Kanri, VSM, studies, DSW results, and quality tools.

Usage:
    python manage.py seed_nlp_demo --user nlp_tmp
    python manage.py seed_nlp_demo --user nlp_tmp --clean
"""

import json
import uuid
from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from agents_api.models import (
    FMEA,
    A3Report,
    ActionItem,
    AnnualObjective,
    DSWResult,
    FMEARow,
    HoshinKPI,
    HoshinProject,
    RCASession,
    Site,
    StrategicObjective,
    ValueStreamMap,
    XMatrixCorrelation,
)
from core.models.hypothesis import Evidence, EvidenceLink, Hypothesis
from core.models.project import Project
from core.models.tenant import Membership, Tenant

User = get_user_model()

# Sentinel for cleanup
NLP_TENANT_SLUG = "next-level-partners"
NLP_TAG = "nlp-demo"


class Command(BaseCommand):
    help = "Seed enterprise demo for Next Level Partners"

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            type=str,
            default="nlp_tmp",
            help="Username for the demo account (default: nlp_tmp)",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Remove all NLP demo data first",
        )

    def handle(self, *args, **options):
        username = options["user"]
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"User '{username}' not found."))
            return

        if options["clean"]:
            self._clean(user)

        # Check for existing tenant
        if Tenant.objects.filter(slug=NLP_TENANT_SLUG).exists():
            self.stdout.write(self.style.WARNING("NLP tenant already exists. Use --clean to recreate."))
            return

        # Ensure enterprise tier
        if user.tier != "enterprise":
            user.tier = "enterprise"
            user.save(update_fields=["tier"])
            self.stdout.write("Set user tier to enterprise")

        # =====================================================================
        # 1. Tenant + Membership
        # =====================================================================
        tenant = Tenant.objects.create(
            name="Next Level Partners",
            slug=NLP_TENANT_SLUG,
            plan=Tenant.Plan.ENTERPRISE,
            max_members=25,
        )
        Membership.objects.create(
            tenant=tenant,
            user=user,
            role=Membership.Role.OWNER,
            joined_at=timezone.now(),
        )
        self.stdout.write(self.style.SUCCESS(f"Created tenant: {tenant.name}"))

        # =====================================================================
        # 2. Site
        # =====================================================================
        site = Site.objects.create(
            tenant=tenant,
            name="Fort Worth Plant",
            code="FW-01",
            business_unit="Packaging",
            plant_manager="Robert Chen",
            ci_leader="Sarah Martinez",
            controller="Karen Phillips",
            address="2100 Industrial Blvd, Fort Worth, TX 76106",
        )
        self.stdout.write(self.style.SUCCESS(f"Created site: {site.name}"))

        # =====================================================================
        # 3. Strategic Objectives (3-5 year)
        # =====================================================================
        so1 = StrategicObjective.objects.create(
            tenant=tenant,
            title="Reduce operational waste by 40% by 2028",
            description="Drive scrap, rework, and material waste down across all packaging lines through systematic DMAIC projects and SPC implementation.",
            owner_name="Robert Chen",
            start_year=2025,
            end_year=2028,
            target_metric="waste_pct",
            target_value=Decimal("40.00"),
            target_unit="% reduction",
            status=StrategicObjective.Status.ACTIVE,
            sort_order=0,
        )
        so2 = StrategicObjective.objects.create(
            tenant=tenant,
            title="Achieve world-class OEE across all lines",
            description="Systematic elimination of the six big losses through TPM, SMED, and data-driven maintenance. Target: 85%+ OEE on all primary lines.",
            owner_name="Robert Chen",
            start_year=2025,
            end_year=2028,
            target_metric="oee",
            target_value=Decimal("85.00"),
            target_unit="%",
            status=StrategicObjective.Status.ACTIVE,
            sort_order=1,
        )
        so3 = StrategicObjective.objects.create(
            tenant=tenant,
            title="Zero critical customer quality escapes",
            description="Eliminate customer-facing quality escapes through upstream SPC, FMEA-driven control plans, and automated inspection.",
            owner_name="Karen Phillips",
            start_year=2025,
            end_year=2028,
            target_metric="defect_rate",
            target_value=Decimal("50.00"),
            target_unit="ppm",
            status=StrategicObjective.Status.ACTIVE,
            sort_order=2,
        )
        self.stdout.write(self.style.SUCCESS("Created 3 strategic objectives"))

        # =====================================================================
        # 4. Annual Objectives (FY2026)
        # =====================================================================
        ao1 = AnnualObjective.objects.create(
            tenant=tenant,
            strategic_objective=so1,
            site=site,
            fiscal_year=2026,
            title="Reduce packaging line scrap from 8% to 4%",
            description="Focus on Line A seal failures and film waste. Primary lever: tension control and changeover standardization.",
            owner_name="Sarah Martinez",
            target_value=Decimal("4.00"),
            actual_value=Decimal("5.80"),
            target_unit="%",
            status=AnnualObjective.Status.AT_RISK,
            sort_order=0,
        )
        ao2 = AnnualObjective.objects.create(
            tenant=tenant,
            strategic_objective=so2,
            site=site,
            fiscal_year=2026,
            title="Improve Line A OEE from 62% to 75%",
            description="Major losses: changeover time (28%), minor stops (15%). SMED and autonomous maintenance focus.",
            owner_name="Sarah Martinez",
            target_value=Decimal("75.00"),
            actual_value=Decimal("68.20"),
            target_unit="%",
            status=AnnualObjective.Status.ON_TRACK,
            sort_order=1,
        )
        ao3 = AnnualObjective.objects.create(
            tenant=tenant,
            strategic_objective=so3,
            site=site,
            fiscal_year=2026,
            title="Implement SPC on top 5 CTQ characteristics",
            description="Deploy real-time SPC monitoring on seal strength, film thickness, label placement, color density, and package weight.",
            owner_name="David Park",
            target_value=Decimal("5.00"),
            actual_value=Decimal("2.00"),
            target_unit="CTQs monitored",
            status=AnnualObjective.Status.ON_TRACK,
            sort_order=2,
        )
        self.stdout.write(self.style.SUCCESS("Created 3 annual objectives"))

        # =====================================================================
        # 5. Hoshin KPIs
        # =====================================================================
        kpi1 = HoshinKPI.objects.create(
            tenant=tenant,
            fiscal_year=2026,
            name="Scrap Rate",
            description="Overall packaging line scrap as % of production",
            target_value=Decimal("4.00"),
            actual_value=Decimal("5.80"),
            unit="%",
            frequency="monthly",
            direction="down",
            aggregation="weighted_avg",
            calculator_result_type="scrap_rate",
            sort_order=0,
        )
        kpi2 = HoshinKPI.objects.create(
            tenant=tenant,
            fiscal_year=2026,
            name="OEE — Line A",
            description="Overall Equipment Effectiveness for primary packaging line",
            target_value=Decimal("75.00"),
            actual_value=Decimal("68.20"),
            unit="%",
            frequency="monthly",
            direction="up",
            aggregation="weighted_avg",
            calculator_result_type="oee",
            sort_order=1,
        )
        kpi3 = HoshinKPI.objects.create(
            tenant=tenant,
            fiscal_year=2026,
            name="First Pass Yield",
            description="Percentage of product passing quality check on first attempt",
            target_value=Decimal("96.00"),
            actual_value=Decimal("93.10"),
            unit="%",
            frequency="monthly",
            direction="up",
            aggregation="weighted_avg",
            calculator_result_type="first_pass_yield",
            sort_order=2,
        )
        kpi4 = HoshinKPI.objects.create(
            tenant=tenant,
            fiscal_year=2026,
            name="Dollar Savings YTD",
            description="Cumulative cost savings from CI projects",
            target_value=Decimal("180000.00"),
            actual_value=Decimal("95000.00"),
            unit="$",
            frequency="monthly",
            direction="up",
            aggregation="sum",
            calculator_result_type="dollar_savings",
            sort_order=3,
        )
        self.stdout.write(self.style.SUCCESS("Created 4 KPIs"))

        # =====================================================================
        # 6. Project A: Packaging Line Scrap Reduction (DMAIC, ANALYZE)
        # =====================================================================
        now = timezone.now()
        proj_a = Project.objects.create(
            tenant=tenant,
            title="Packaging Line Scrap Reduction — Line A",
            status=Project.Status.ACTIVE,
            methodology=Project.Methodology.DMAIC,
            current_phase=Project.Phase.ANALYZE,
            phase_history=[
                {
                    "phase": "define",
                    "entered_at": (now - timedelta(days=45)).isoformat(),
                    "notes": "Problem scoped to Line A seal failures and film waste",
                },
                {
                    "phase": "measure",
                    "entered_at": (now - timedelta(days=30)).isoformat(),
                    "notes": "MSA passed, baseline data collected (8.2% scrap)",
                },
                {
                    "phase": "analyze",
                    "entered_at": (now - timedelta(days=14)).isoformat(),
                    "notes": "Hypotheses formed, statistical analysis underway",
                },
            ],
            problem_whats=[
                "Seal failure rate on Line A packaging exceeds target by 4.2 percentage points",
                "Film waste during changeover averaging 12% of roll length",
            ],
            problem_wheres=[
                "Line A — primary packaging line, 3-shift operation",
                "Seal station (heat bar #2) and unwind station",
            ],
            problem_whens=[
                "Elevated since Q4 2025 when new film supplier was qualified",
                "Worst during first 30 minutes after changeover (22% vs 6% steady-state)",
            ],
            problem_magnitude=(
                "Scrap rate 8.2% vs 4.0% target. Annual waste: $420K at current rates. "
                "Customer complaints up 35% YoY on seal integrity."
            ),
            problem_trend=Project.Trend.STABLE,
            problem_since="2025-10-15 (new film supplier qualification)",
            problem_statement=(
                "Packaging Line A scrap rate has been running at 8.2% against a 4.0% target "
                "since Q4 2025, driven by seal failures and changeover film waste. "
                "Annual cost of excess scrap is approximately $420K. The problem has stabilized "
                "but not improved despite initial countermeasures."
            ),
            impact_financial="$420K/year in excess scrap. $180K savings target for FY2026.",
            impact_customer="35% increase in seal-integrity complaints. 2 major accounts have issued formal corrective action requests.",
            impact_quality="Cpk on seal strength dropped from 1.45 to 1.12. First pass yield at 93.1% vs 96% target.",
            impact_delivery="Rework queue adding 0.5 days to lead time on affected SKUs.",
            goal_statement="Reduce Line A scrap rate from 8.2% to 4.0% or below within 12 weeks and sustain for 60 days.",
            goal_metric="Scrap Rate (%)",
            goal_baseline="8.2%",
            goal_target="4.0%",
            goal_unit="%",
            goal_deadline=date.today() + timedelta(days=60),
            scope_in=[
                "Line A seal station and unwind station",
                "Film supplier qualification data",
                "Changeover procedures and SOP compliance",
                "Heat bar temperature profiles",
            ],
            scope_out=[
                "Lines B and C (different product families)",
                "Upstream printing defects (separate project)",
                "Capital equipment replacement",
            ],
            constraints=[
                "Cannot take Line A offline for more than 4 hours (24/7 production)",
                "Must maintain current throughput rate (180 packages/min)",
                "Film supplier change requires 90-day qualification cycle",
            ],
            assumptions=[
                "Gage R&R on seal strength tester is adequate (passed at 12.4%)",
                "Film spec from new supplier matches old supplier certification",
                "Operators are following current SOPs as written",
            ],
            champion_name="Robert Chen",
            champion_title="Plant Manager",
            leader_name="Sarah Martinez",
            leader_title="CI Leader / Black Belt",
            team_members=[
                {"name": "David Park", "role": "Quality Engineer", "department": "Quality"},
                {"name": "Maria Gonzalez", "role": "Line A Supervisor", "department": "Production"},
                {"name": "James Wu", "role": "Maintenance Tech Lead", "department": "Maintenance"},
                {"name": "Lisa Chen", "role": "Process Engineer", "department": "Engineering"},
            ],
            target_completion=date.today() + timedelta(days=60),
            milestones=[
                {
                    "name": "Define phase complete",
                    "target_date": (date.today() - timedelta(days=45)).isoformat(),
                    "actual_date": (date.today() - timedelta(days=44)).isoformat(),
                    "status": "completed",
                },
                {
                    "name": "MSA validation",
                    "target_date": (date.today() - timedelta(days=35)).isoformat(),
                    "actual_date": (date.today() - timedelta(days=33)).isoformat(),
                    "status": "completed",
                },
                {
                    "name": "Baseline data collected",
                    "target_date": (date.today() - timedelta(days=28)).isoformat(),
                    "actual_date": (date.today() - timedelta(days=28)).isoformat(),
                    "status": "completed",
                },
                {
                    "name": "Root cause confirmed",
                    "target_date": (date.today() + timedelta(days=7)).isoformat(),
                    "actual_date": None,
                    "status": "in_progress",
                },
                {
                    "name": "Countermeasures implemented",
                    "target_date": (date.today() + timedelta(days=30)).isoformat(),
                    "actual_date": None,
                    "status": "planned",
                },
                {
                    "name": "60-day sustain verification",
                    "target_date": (date.today() + timedelta(days=90)).isoformat(),
                    "actual_date": None,
                    "status": "planned",
                },
            ],
            domain="manufacturing",
            can_experiment=True,
            tags=[NLP_TAG, "packaging", "scrap-reduction", "DMAIC", "seal-failure"],
        )

        hp_a = HoshinProject.objects.create(
            project=proj_a,
            site=site,
            project_class=HoshinProject.ProjectClass.PROJECT,
            project_type=HoshinProject.ProjectType.QUALITY,
            opportunity=HoshinProject.Opportunity.BUDGETED_NEW,
            hoshin_status=HoshinProject.HoshinStatus.ACTIVE,
            fiscal_year=2026,
            annual_savings_target=Decimal("180000.00"),
            calculation_method="waste_pct",
            monthly_actuals=[
                {
                    "month": "2026-01",
                    "baseline": 8.2,
                    "actual": 8.0,
                    "volume": 540000,
                    "cost_per_unit": 0.087,
                    "savings": 940,
                },
                {
                    "month": "2026-02",
                    "baseline": 8.2,
                    "actual": 7.6,
                    "volume": 510000,
                    "cost_per_unit": 0.087,
                    "savings": 2660,
                },
                {
                    "month": "2026-03",
                    "baseline": 8.2,
                    "actual": 7.1,
                    "volume": 560000,
                    "cost_per_unit": 0.087,
                    "savings": 5350,
                },
                {
                    "month": "2026-04",
                    "baseline": 8.2,
                    "actual": 6.5,
                    "volume": 530000,
                    "cost_per_unit": 0.087,
                    "savings": 7830,
                },
                {
                    "month": "2026-05",
                    "baseline": 8.2,
                    "actual": 6.1,
                    "volume": 550000,
                    "cost_per_unit": 0.087,
                    "savings": 10060,
                },
                {
                    "month": "2026-06",
                    "baseline": 8.2,
                    "actual": 5.8,
                    "volume": 540000,
                    "cost_per_unit": 0.087,
                    "savings": 11270,
                },
            ],
        )

        # --- Hypotheses for Project A ---
        h1 = Hypothesis.objects.create(
            project=proj_a,
            statement="If film tension variability during unwind exceeds ±5%, then seal failures will increase because inconsistent film presentation to the heat bar causes uneven seal formation.",
            if_clause="film tension variability during unwind exceeds ±5%",
            then_clause="seal failures will increase",
            because_clause="inconsistent film presentation to the heat bar causes uneven seal formation",
            independent_variable="Film Tension CV",
            independent_var_values=["Current: 8.3% CV", "Target: <5% CV"],
            dependent_variable="Seal Failure Rate",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.DECREASE,
            predicted_magnitude="Expect 40-60% reduction in seal failures if tension CV < 5%",
            rationale="SPC charts show tension spikes correlating with seal failure clusters. New film has different elastic modulus requiring tension re-optimization.",
            test_method="1. Characterize tension variability via high-speed data logger. 2. DOE on tension setpoint and dancer pressure. 3. Confirmation run at optimal settings.",
            success_criteria="Confirm: Tension CV >5% AND DOE shows significant main effect on seal strength. Refute: Tension CV within spec or no correlation.",
            prior_probability=0.50,
            current_probability=0.82,
            confirmation_threshold=0.90,
            rejection_threshold=0.10,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )
        h2 = Hypothesis.objects.create(
            project=proj_a,
            statement="If ambient humidity exceeds 60% RH, then adhesive delamination will increase because the water-based adhesive in the new film absorbs moisture and loses bond strength.",
            if_clause="ambient humidity exceeds 60% RH",
            then_clause="adhesive delamination will increase",
            because_clause="the water-based adhesive in the new film absorbs moisture and loses bond strength before heat activation",
            independent_variable="Ambient Humidity",
            independent_var_values=["Current: 45-65% RH", "Controlled: <55% RH"],
            dependent_variable="Delamination Rate",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.DECREASE,
            predicted_magnitude="Expect >50% reduction if humidity consistently <55%",
            rationale="New film supplier uses water-based adhesive vs old supplier's solvent-based. Plant HVAC has limited humidity control.",
            test_method="1. Correlate humidity logs with hourly defect rates. 2. Test seal strength at controlled humidity levels in lab.",
            success_criteria="Confirm: Statistically significant correlation (p<0.05) between humidity >60% and delamination. Refute: No correlation.",
            prior_probability=0.50,
            current_probability=0.35,
            confirmation_threshold=0.90,
            rejection_threshold=0.10,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )
        h3 = Hypothesis.objects.create(
            project=proj_a,
            statement="If changeover procedure inconsistency drives first-hour scrap spikes, then standardizing the changeover SOP and adding a verification checklist will eliminate the startup quality gap.",
            if_clause="changeover procedure inconsistency drives first-hour scrap spikes",
            then_clause="standardizing the changeover SOP will eliminate the startup quality gap",
            because_clause="operators are using different heat bar warm-up sequences and tension presets, leading to variable initial conditions",
            independent_variable="Changeover Procedure Compliance",
            independent_var_values=["Current: ~60% compliance", "Target: 95%+ compliance"],
            dependent_variable="First-Hour Scrap Rate",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.DECREASE,
            predicted_magnitude="Expect first-hour scrap to drop from 22% to <8% with standardized procedure",
            rationale="Time-series analysis shows 22% scrap in first 30 min post-changeover vs 6% steady-state. Operator interviews reveal 3 different warm-up sequences in use.",
            test_method="1. Shadow 10 changeovers across all shifts and document actual steps. 2. Develop standardized SOP. 3. Pilot on shift A for 2 weeks.",
            success_criteria="Confirm: Significant difference between operator procedures AND pilot shows first-hour scrap <10%. Refute: All operators follow same sequence.",
            prior_probability=0.50,
            current_probability=0.71,
            confirmation_threshold=0.90,
            rejection_threshold=0.10,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )

        # --- Evidence for Project A ---
        e1 = Evidence.objects.create(
            project=proj_a,
            summary="SPC analysis: film tension CV measured at 8.3% (spec <5%). Tension spikes correlate with 73% of seal failure events in 2-week study.",
            details="High-speed data logger on unwind station recorded 14,400 tension readings over 2 weeks. CV = 8.3% (USL = 5%). Cross-referenced with quality inspection timestamps: 73% of seal failures (n=47/64) occurred within 30 seconds of a tension excursion >2σ.",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="SPC analysis — tension variability vs seal failures",
            result_type=Evidence.ResultType.STATISTICAL,
            confidence=0.90,
            p_value=0.001,
            effect_size=0.73,
            sample_size=14400,
            statistical_test="Chi-squared test of independence + correlation analysis",
            created_by=user,
        )
        e2 = Evidence.objects.create(
            project=proj_a,
            summary="DOE on seal temperature, pressure, and dwell time: temperature × pressure interaction significant (p=0.008). Optimal: 185°C, 45 PSI, 0.8s dwell.",
            details="2³ full factorial DOE with 4 center points. Factors: temperature (175-195°C), pressure (35-55 PSI), dwell time (0.6-1.0s). Temperature main effect: p=0.002. Pressure main effect: p=0.015. Temp×Pressure interaction: p=0.008. Dwell time: not significant (p=0.34). Response surface shows optimal at 185°C, 45 PSI. Confirmation run: 0/50 seal failures at optimal.",
            source_type=Evidence.SourceType.EXPERIMENT,
            source_description="2³ factorial DOE on sealing parameters",
            result_type=Evidence.ResultType.STATISTICAL,
            confidence=0.92,
            p_value=0.002,
            effect_size=1.4,
            sample_size=36,
            statistical_test="Full factorial ANOVA with center points",
            created_by=user,
        )
        e3 = Evidence.objects.create(
            project=proj_a,
            summary="Changeover observation: 10 changeovers shadowed across 3 shifts. Found 3 distinct warm-up sequences, 22% first-hour scrap vs 6% steady-state.",
            details="Documented actual operator behavior during 10 changeovers (4 on A shift, 3 on B, 3 on C). Key findings: (1) Three different heat bar warm-up sequences observed. (2) A-shift operators wait for temperature stabilization; B/C shifts start immediately. (3) First-hour scrap 22% ± 4% vs steady-state 6% ± 1.2%. (4) No written SOP exists for warm-up sequence — tribal knowledge only.",
            source_type=Evidence.SourceType.OBSERVATION,
            source_description="Direct observation of 10 changeover events",
            result_type=Evidence.ResultType.QUALITATIVE,
            confidence=0.85,
            created_by=user,
        )
        e4 = Evidence.objects.create(
            project=proj_a,
            summary="Humidity correlation: regression analysis shows no significant relationship between ambient humidity (range 42-63% RH) and seal failure rate (p=0.41, R²=0.02).",
            details="30 days of hourly humidity logs (720 data points) correlated with hourly seal failure counts. Simple linear regression: R²=0.02, p=0.41. Segmented analysis at >55% and >60% thresholds also non-significant. Conclusion: humidity is not a driver in the observed range.",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="Regression analysis — humidity vs seal failure rate",
            result_type=Evidence.ResultType.STATISTICAL,
            confidence=0.88,
            p_value=0.41,
            effect_size=0.02,
            sample_size=720,
            statistical_test="Simple linear regression + segmented analysis",
            created_by=user,
        )
        e5 = Evidence.objects.create(
            project=proj_a,
            summary="Gage R&R on seal strength tester: %GRR = 12.4% of tolerance (acceptable). 3 operators, 10 parts, 3 trials.",
            details="Crossed Gage R&R study per AIAG MSA 4th edition. 3 operators × 10 parts × 3 trials = 90 measurements. %GRR = 12.4% of tolerance (spec: 8-15 N). Repeatability: 8.1%. Reproducibility: 9.3%. Number of distinct categories (ndc) = 5. Measurement system is acceptable for process monitoring and DMAIC analysis.",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="Gage R&R study — AIAG MSA method",
            result_type=Evidence.ResultType.STATISTICAL,
            confidence=0.93,
            sample_size=90,
            statistical_test="Crossed Gage R&R (ANOVA method)",
            created_by=user,
        )

        # --- Evidence Links with Bayesian updates ---
        # E1 (tension spikes) → strongly supports H1
        link = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e1,
            likelihood_ratio=6.0,
            reasoning="Tension CV is 66% above spec limit. 73% of seal failures correlate with tension excursions. Direct evidence supporting the tension variability hypothesis.",
        )
        h1.apply_evidence(link)
        # E2 (DOE) → moderately supports H1, neutral to H3
        link = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e2,
            likelihood_ratio=3.0,
            reasoning="DOE confirms seal parameters interact — but the root cause may be tension affecting the seal interface rather than wrong setpoints. Moderate support.",
        )
        h1.apply_evidence(link)
        # E3 (changeover observation) → supports H3
        link = EvidenceLink.objects.create(
            hypothesis=h3,
            evidence=e3,
            likelihood_ratio=4.5,
            reasoning="Three different procedures, no SOP, massive first-hour quality gap. Strong evidence that changeover inconsistency is a real driver.",
        )
        h3.apply_evidence(link)
        # E3 → mildly supports H1 (tension may not be set correctly post-changeover)
        link = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e3,
            likelihood_ratio=1.3,
            reasoning="If operators use different warm-up sequences, tension settings may also vary post-changeover. Mild support for tension hypothesis.",
        )
        h1.apply_evidence(link)
        # E4 (humidity non-significant) → strongly opposes H2
        link = EvidenceLink.objects.create(
            hypothesis=h2,
            evidence=e4,
            likelihood_ratio=0.15,
            reasoning="No statistical relationship between humidity and seal failures across the observed range. Effectively rules out humidity as a contributing factor.",
        )
        h2.apply_evidence(link)
        # E5 (Gage R&R) → neutral to H1, slight support by confirming measurement adequacy
        link = EvidenceLink.objects.create(
            hypothesis=h1,
            evidence=e5,
            likelihood_ratio=1.1,
            reasoning="Confirms measurement system is adequate (12.4% GRR). Our data-driven conclusions about tension and seal strength can be trusted.",
        )
        h1.apply_evidence(link)

        h1.refresh_from_db()
        h2.refresh_from_db()
        h3.refresh_from_db()
        self.stdout.write(
            self.style.SUCCESS(
                f"Project A: H1={h1.current_probability:.0%}, H2={h2.current_probability:.0%}, H3={h3.current_probability:.0%}"
            )
        )

        # --- Action Items for Project A ---
        ActionItem.objects.create(
            project=proj_a,
            title="Install real-time tension monitoring on unwind station",
            description="Deploy high-speed tension sensor with SPC alarming. Integrate with SCADA for automatic alerts when CV exceeds 5%.",
            owner_name="James Wu",
            status=ActionItem.Status.IN_PROGRESS,
            start_date=date.today() - timedelta(days=7),
            due_date=date.today() + timedelta(days=14),
            progress=60,
            sort_order=0,
            source_type="hoshin",
            source_id=hp_a.id,
        )
        ActionItem.objects.create(
            project=proj_a,
            title="Develop standardized changeover SOP with verification checklist",
            description="Create single standard changeover procedure. Include heat bar warm-up sequence, tension preset verification, and first-article inspection gate.",
            owner_name="Sarah Martinez",
            status=ActionItem.Status.IN_PROGRESS,
            start_date=date.today() - timedelta(days=3),
            due_date=date.today() + timedelta(days=10),
            progress=30,
            sort_order=1,
            source_type="hoshin",
            source_id=hp_a.id,
        )
        ActionItem.objects.create(
            project=proj_a,
            title="Run confirmation DOE at optimized seal parameters",
            description="Full factorial at 185°C/45PSI/0.8s with 50-piece confirmation run. Success: 0 seal failures.",
            owner_name="Lisa Chen",
            status=ActionItem.Status.NOT_STARTED,
            due_date=date.today() + timedelta(days=21),
            sort_order=2,
            source_type="hoshin",
            source_id=hp_a.id,
        )
        ActionItem.objects.create(
            project=proj_a,
            title="Train all operators on new changeover SOP",
            description="Hands-on training for all 3 shifts. Include TWI Job Instruction method. Verify with observed changeovers.",
            owner_name="Maria Gonzalez",
            status=ActionItem.Status.NOT_STARTED,
            due_date=date.today() + timedelta(days=28),
            sort_order=3,
            source_type="hoshin",
            source_id=hp_a.id,
        )
        self.stdout.write(self.style.SUCCESS("Created Project A with 3 hypotheses, 5 evidence, 4 action items"))

        # =====================================================================
        # 7. Project B: OEE Improvement (DMAIC, MEASURE)
        # =====================================================================
        proj_b = Project.objects.create(
            tenant=tenant,
            title="OEE Improvement — Line A",
            status=Project.Status.ACTIVE,
            methodology=Project.Methodology.DMAIC,
            current_phase=Project.Phase.MEASURE,
            phase_history=[
                {
                    "phase": "define",
                    "entered_at": (now - timedelta(days=21)).isoformat(),
                    "notes": "Problem scoped: OEE at 62%, target 75%",
                },
                {
                    "phase": "measure",
                    "entered_at": (now - timedelta(days=10)).isoformat(),
                    "notes": "Automated OEE data collection deployed, loss categorization underway",
                },
            ],
            problem_whats=[
                "Line A OEE at 62% against 75% target",
                "Major losses: changeover (28% of downtime), minor stops (15%), speed losses (12%)",
            ],
            problem_wheres=["Line A — all stations, particularly slitter and laminator"],
            problem_whens=["Persistent issue — worsened after product mix change in Q3 2025"],
            problem_magnitude="13 percentage points below target. Estimated lost capacity: $120K/year equivalent.",
            problem_trend=Project.Trend.DECREASING,
            problem_since="2025-07 (product mix change)",
            problem_statement="Line A OEE has been running at 62% against a 75% target, primarily driven by changeover losses (28%), minor stops (15%), and speed losses (12%).",
            impact_financial="$120K/year in lost capacity. Additional overtime costs of ~$35K/year to meet demand.",
            goal_statement="Improve Line A OEE from 62% to 75% within 16 weeks.",
            goal_metric="OEE (%)",
            goal_baseline="62%",
            goal_target="75%",
            goal_unit="%",
            goal_deadline=date.today() + timedelta(days=90),
            champion_name="Robert Chen",
            champion_title="Plant Manager",
            leader_name="Sarah Martinez",
            leader_title="CI Leader",
            team_members=[
                {"name": "Maria Gonzalez", "role": "Line Supervisor", "department": "Production"},
                {"name": "James Wu", "role": "Maintenance Tech Lead", "department": "Maintenance"},
            ],
            domain="manufacturing",
            can_experiment=True,
            tags=[NLP_TAG, "OEE", "DMAIC", "changeover", "SMED"],
        )

        hp_b = HoshinProject.objects.create(
            project=proj_b,
            site=site,
            project_class=HoshinProject.ProjectClass.PROJECT,
            project_type=HoshinProject.ProjectType.THROUGHPUT,
            opportunity=HoshinProject.Opportunity.BUDGETED_NEW,
            hoshin_status=HoshinProject.HoshinStatus.ACTIVE,
            fiscal_year=2026,
            annual_savings_target=Decimal("120000.00"),
            calculation_method="time_reduction",
            monthly_actuals=[
                {
                    "month": "2026-01",
                    "baseline": 62.0,
                    "actual": 63.5,
                    "volume": 480,
                    "cost_per_unit": 250,
                    "savings": 1800,
                },
                {
                    "month": "2026-02",
                    "baseline": 62.0,
                    "actual": 65.1,
                    "volume": 460,
                    "cost_per_unit": 250,
                    "savings": 3570,
                },
                {
                    "month": "2026-03",
                    "baseline": 62.0,
                    "actual": 66.8,
                    "volume": 500,
                    "cost_per_unit": 250,
                    "savings": 6000,
                },
                {
                    "month": "2026-04",
                    "baseline": 62.0,
                    "actual": 67.5,
                    "volume": 490,
                    "cost_per_unit": 250,
                    "savings": 6740,
                },
                {
                    "month": "2026-05",
                    "baseline": 62.0,
                    "actual": 68.0,
                    "volume": 510,
                    "cost_per_unit": 250,
                    "savings": 7650,
                },
                {
                    "month": "2026-06",
                    "baseline": 62.0,
                    "actual": 68.2,
                    "volume": 500,
                    "cost_per_unit": 250,
                    "savings": 7750,
                },
            ],
        )

        # Hypotheses for Project B
        hb1 = Hypothesis.objects.create(
            project=proj_b,
            statement="If changeover time on the slitter is reduced from 45 to 20 minutes via SMED, then availability will increase by 8 percentage points.",
            if_clause="changeover time on the slitter is reduced from 45 to 20 minutes via SMED",
            then_clause="availability will increase by 8 percentage points",
            because_clause="slitter changeovers occur 3x/shift and currently consume 28% of available time",
            independent_variable="Changeover Time",
            dependent_variable="OEE Availability",
            dependent_var_unit="%",
            predicted_direction=Hypothesis.Direction.INCREASE,
            prior_probability=0.60,
            current_probability=0.60,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )
        hb2 = Hypothesis.objects.create(
            project=proj_b,
            statement="If autonomous maintenance routines are implemented, then minor stops will decrease by 50% because operators will catch early warning signs before failures occur.",
            if_clause="autonomous maintenance routines are implemented",
            then_clause="minor stops will decrease by 50%",
            because_clause="operators will catch early warning signs before failures occur through daily inspection checklists",
            independent_variable="AM Program Implementation",
            dependent_variable="Minor Stop Frequency",
            dependent_var_unit="stops/shift",
            predicted_direction=Hypothesis.Direction.DECREASE,
            prior_probability=0.50,
            current_probability=0.50,
            status=Hypothesis.Status.ACTIVE,
            created_by=user,
        )

        # Evidence for Project B
        Evidence.objects.create(
            project=proj_b,
            summary="Pareto analysis of downtime: changeover accounts for 28% (135 min/shift avg), minor stops 15% (72 min/shift), speed losses 12% (58 min/shift).",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="OEE loss categorization — 30-day automated data collection",
            result_type=Evidence.ResultType.QUANTITATIVE,
            confidence=0.90,
            sample_size=90,
            created_by=user,
        )
        eb2 = Evidence.objects.create(
            project=proj_b,
            summary="SMED analysis: 67% of slitter changeover steps are external work currently done internally. Estimated reduction: 45min → 18min.",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="SMED video analysis of 5 changeover events",
            result_type=Evidence.ResultType.QUANTITATIVE,
            confidence=0.85,
            sample_size=5,
            created_by=user,
        )
        eb3 = Evidence.objects.create(
            project=proj_b,
            summary="Minor stop analysis: top 3 causes are film splice jams (38%), sensor misalignment (27%), and label feed jams (19%). All are detectable through routine inspection.",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description="Pareto analysis of minor stop root causes",
            result_type=Evidence.ResultType.CATEGORICAL,
            confidence=0.82,
            sample_size=156,
            created_by=user,
        )

        link = EvidenceLink.objects.create(
            hypothesis=hb1,
            evidence=eb2,
            likelihood_ratio=3.5,
            reasoning="SMED analysis shows clear potential for 60% changeover reduction. 67% external work is actionable.",
        )
        hb1.apply_evidence(link)
        link = EvidenceLink.objects.create(
            hypothesis=hb2,
            evidence=eb3,
            likelihood_ratio=2.0,
            reasoning="Top causes are all detectable through routine inspection, supporting AM approach.",
        )
        hb2.apply_evidence(link)

        hb1.refresh_from_db()
        hb2.refresh_from_db()
        self.stdout.write(
            self.style.SUCCESS(f"Project B: Hb1={hb1.current_probability:.0%}, Hb2={hb2.current_probability:.0%}")
        )

        # =====================================================================
        # 8. Project C: SMED Kaizen (COMPLETED)
        # =====================================================================
        proj_c = Project.objects.create(
            tenant=tenant,
            title="Quick Changeover (SMED) — Slitter",
            status=Project.Status.RESOLVED,
            methodology=Project.Methodology.PDCA,
            current_phase=Project.Phase.CONTROL,
            phase_history=[
                {
                    "phase": "define",
                    "entered_at": (now - timedelta(days=90)).isoformat(),
                    "notes": "Kaizen event chartered",
                },
                {
                    "phase": "measure",
                    "entered_at": (now - timedelta(days=85)).isoformat(),
                    "notes": "Video analysis complete",
                },
                {
                    "phase": "analyze",
                    "entered_at": (now - timedelta(days=84)).isoformat(),
                    "notes": "Internal/external work separated",
                },
                {
                    "phase": "improve",
                    "entered_at": (now - timedelta(days=82)).isoformat(),
                    "notes": "New procedure implemented and tested",
                },
                {
                    "phase": "control",
                    "entered_at": (now - timedelta(days=78)).isoformat(),
                    "notes": "Standard work posted, 30-day sustain confirmed",
                },
            ],
            problem_statement="Slitter changeover time averaging 45 minutes. Target: <20 minutes.",
            goal_statement="Reduce slitter changeover from 45 minutes to under 20 minutes in a 5-day kaizen event.",
            goal_metric="Changeover Time (min)",
            goal_baseline="45",
            goal_target="20",
            goal_unit="minutes",
            goal_deadline=date.today() - timedelta(days=78),
            champion_name="Robert Chen",
            leader_name="James Wu",
            team_members=[
                {"name": "Maria Gonzalez", "role": "Line Supervisor", "department": "Production"},
                {"name": "Tom Bradley", "role": "Slitter Operator", "department": "Production"},
                {"name": "Sarah Martinez", "role": "CI Leader", "department": "CI"},
            ],
            resolution_summary="Changeover reduced from 45 to 18 minutes (60% reduction). Key actions: externalized blade prep, standardized settings card, quick-release fixtures. Sustained for 30 days.",
            domain="manufacturing",
            tags=[NLP_TAG, "SMED", "kaizen", "changeover", "slitter"],
        )

        hp_c = HoshinProject.objects.create(
            project=proj_c,
            site=site,
            project_class=HoshinProject.ProjectClass.KAIZEN,
            project_type=HoshinProject.ProjectType.LABOR,
            opportunity=HoshinProject.Opportunity.BUDGETED_NEW,
            hoshin_status=HoshinProject.HoshinStatus.COMPLETED,
            fiscal_year=2026,
            annual_savings_target=Decimal("45000.00"),
            calculation_method="time_reduction",
            kaizen_charter={
                "event_date": (date.today() - timedelta(days=90)).isoformat(),
                "end_date": (date.today() - timedelta(days=85)).isoformat(),
                "location": "Fort Worth Plant — Line A, Slitter Station",
                "event_type": "SMED",
                "primary_metric": "Changeover Time",
                "primary_baseline": "45 min",
                "primary_target": "20 min",
                "secondary_metric": "First-piece quality",
                "secondary_baseline": "85%",
                "secondary_target": "95%",
                "process_start": "Slitter shutdown signal",
                "process_end": "First good piece at rate",
            },
        )

        # Completed action item
        ActionItem.objects.create(
            project=proj_c,
            title="Post standard work at slitter station",
            owner_name="James Wu",
            status=ActionItem.Status.COMPLETED,
            start_date=date.today() - timedelta(days=82),
            end_date=date.today() - timedelta(days=80),
            due_date=date.today() - timedelta(days=80),
            progress=100,
            sort_order=0,
            source_type="hoshin",
            source_id=hp_c.id,
        )
        self.stdout.write(self.style.SUCCESS("Created Project C (SMED kaizen — completed)"))

        # =====================================================================
        # 9. Value Stream Maps
        # =====================================================================
        step_ids = [str(uuid.uuid4()) for _ in range(5)]
        inv_ids = [str(uuid.uuid4()) for _ in range(4)]
        burst_ids = [str(uuid.uuid4()) for _ in range(3)]

        vsm_current = ValueStreamMap.objects.create(
            owner=user,
            project=proj_b,
            name="Packaging Line A — Current State",
            status=ValueStreamMap.Status.CURRENT,
            fiscal_year="2026",
            product_family="Standard Flexible Packaging",
            customer_name="National Distributors",
            customer_demand="460 units/day",
            takt_time=62.6,
            supplier_name="Pacific Films Inc.",
            supply_frequency="Weekly (Tuesdays)",
            process_steps=[
                {
                    "id": step_ids[0],
                    "name": "Receiving & Inspection",
                    "x": 150,
                    "y": 300,
                    "cycle_time": 45,
                    "changeover_time": 0,
                    "uptime": 98,
                    "operators": 1,
                    "shifts": 1,
                    "batch_size": 1,
                },
                {
                    "id": step_ids[1],
                    "name": "Slitting",
                    "x": 350,
                    "y": 300,
                    "cycle_time": 180,
                    "changeover_time": 18,
                    "uptime": 85,
                    "operators": 1,
                    "shifts": 3,
                    "batch_size": 1,
                },
                {
                    "id": step_ids[2],
                    "name": "Printing",
                    "x": 550,
                    "y": 300,
                    "cycle_time": 240,
                    "changeover_time": 35,
                    "uptime": 82,
                    "operators": 2,
                    "shifts": 3,
                    "batch_size": 1,
                },
                {
                    "id": step_ids[3],
                    "name": "Laminating",
                    "x": 750,
                    "y": 300,
                    "cycle_time": 195,
                    "changeover_time": 25,
                    "uptime": 88,
                    "operators": 1,
                    "shifts": 3,
                    "batch_size": 1,
                },
                {
                    "id": step_ids[4],
                    "name": "Packaging & Sealing",
                    "x": 950,
                    "y": 300,
                    "cycle_time": 187,
                    "changeover_time": 30,
                    "uptime": 80,
                    "operators": 2,
                    "shifts": 3,
                    "batch_size": 1,
                },
            ],
            inventory=[
                {
                    "id": inv_ids[0],
                    "before_step_id": step_ids[1],
                    "quantity": 8000,
                    "days_of_supply": 3.5,
                    "x": 250,
                    "y": 300,
                },
                {
                    "id": inv_ids[1],
                    "before_step_id": step_ids[2],
                    "quantity": 5000,
                    "days_of_supply": 2.2,
                    "x": 450,
                    "y": 300,
                },
                {
                    "id": inv_ids[2],
                    "before_step_id": step_ids[3],
                    "quantity": 4500,
                    "days_of_supply": 2.0,
                    "x": 650,
                    "y": 300,
                },
                {
                    "id": inv_ids[3],
                    "before_step_id": step_ids[4],
                    "quantity": 6000,
                    "days_of_supply": 2.6,
                    "x": 850,
                    "y": 300,
                },
            ],
            information_flow=[
                {
                    "id": str(uuid.uuid4()),
                    "from_id": "customer",
                    "to_id": "production_control",
                    "type": "electronic",
                    "label": "Daily orders via EDI",
                },
                {
                    "id": str(uuid.uuid4()),
                    "from_id": "production_control",
                    "to_id": step_ids[0],
                    "type": "manual",
                    "label": "Weekly production schedule",
                },
                {
                    "id": str(uuid.uuid4()),
                    "from_id": "production_control",
                    "to_id": "supplier",
                    "type": "electronic",
                    "label": "Weekly PO via EDI",
                },
            ],
            material_flow=[
                {"id": str(uuid.uuid4()), "from_step_id": step_ids[0], "to_step_id": step_ids[1], "type": "push"},
                {"id": str(uuid.uuid4()), "from_step_id": step_ids[1], "to_step_id": step_ids[2], "type": "push"},
                {"id": str(uuid.uuid4()), "from_step_id": step_ids[2], "to_step_id": step_ids[3], "type": "push"},
                {"id": str(uuid.uuid4()), "from_step_id": step_ids[3], "to_step_id": step_ids[4], "type": "push"},
            ],
            kaizen_bursts=[
                {
                    "id": burst_ids[0],
                    "x": 360,
                    "y": 200,
                    "text": "SMED on slitter changeover (18min target)",
                    "priority": "high",
                },
                {
                    "id": burst_ids[1],
                    "x": 560,
                    "y": 200,
                    "text": "Reduce print WIP to 1 day supply",
                    "priority": "medium",
                },
                {"id": burst_ids[2], "x": 960, "y": 200, "text": "Implement SPC on seal strength", "priority": "high"},
            ],
            total_lead_time=12.4,
            total_process_time=847,
            pce=0.08,
        )

        vsm_future = ValueStreamMap.objects.create(
            owner=user,
            project=proj_b,
            name="Packaging Line A — Future State",
            status=ValueStreamMap.Status.FUTURE,
            fiscal_year="2026",
            product_family="Standard Flexible Packaging",
            customer_name="National Distributors",
            customer_demand="460 units/day",
            takt_time=62.6,
            supplier_name="Pacific Films Inc.",
            supply_frequency="2x/week (Tue, Fri)",
            process_steps=[
                {
                    "id": str(uuid.uuid4()),
                    "name": "Receiving",
                    "x": 150,
                    "y": 300,
                    "cycle_time": 30,
                    "changeover_time": 0,
                    "uptime": 99,
                    "operators": 1,
                    "shifts": 1,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Slitting",
                    "x": 350,
                    "y": 300,
                    "cycle_time": 165,
                    "changeover_time": 10,
                    "uptime": 92,
                    "operators": 1,
                    "shifts": 3,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Printing",
                    "x": 550,
                    "y": 300,
                    "cycle_time": 220,
                    "changeover_time": 20,
                    "uptime": 90,
                    "operators": 2,
                    "shifts": 3,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Laminating",
                    "x": 750,
                    "y": 300,
                    "cycle_time": 180,
                    "changeover_time": 15,
                    "uptime": 93,
                    "operators": 1,
                    "shifts": 3,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Packaging & Sealing",
                    "x": 950,
                    "y": 300,
                    "cycle_time": 175,
                    "changeover_time": 15,
                    "uptime": 90,
                    "operators": 2,
                    "shifts": 3,
                },
            ],
            inventory=[
                {
                    "id": str(uuid.uuid4()),
                    "before_step_id": "slitting",
                    "quantity": 2500,
                    "days_of_supply": 1.1,
                    "x": 250,
                    "y": 300,
                },
                {
                    "id": str(uuid.uuid4()),
                    "before_step_id": "printing",
                    "quantity": 1500,
                    "days_of_supply": 0.7,
                    "x": 450,
                    "y": 300,
                },
                {
                    "id": str(uuid.uuid4()),
                    "before_step_id": "laminating",
                    "quantity": 1200,
                    "days_of_supply": 0.5,
                    "x": 650,
                    "y": 300,
                },
                {
                    "id": str(uuid.uuid4()),
                    "before_step_id": "packaging",
                    "quantity": 2000,
                    "days_of_supply": 0.9,
                    "x": 850,
                    "y": 300,
                },
            ],
            material_flow=[
                {"id": str(uuid.uuid4()), "from_step_id": "receiving", "to_step_id": "slitting", "type": "pull"},
                {"id": str(uuid.uuid4()), "from_step_id": "slitting", "to_step_id": "printing", "type": "fifo"},
                {"id": str(uuid.uuid4()), "from_step_id": "printing", "to_step_id": "laminating", "type": "fifo"},
                {"id": str(uuid.uuid4()), "from_step_id": "laminating", "to_step_id": "packaging", "type": "pull"},
            ],
            total_lead_time=4.2,
            total_process_time=770,
            pce=0.21,
        )

        # Pair the VSMs
        vsm_current.paired_with = vsm_future
        vsm_current.save(update_fields=["paired_with"])
        vsm_future.paired_with = vsm_current
        vsm_future.save(update_fields=["paired_with"])

        # Link SMED project to kaizen burst
        hp_c.source_vsm = vsm_current
        hp_c.source_burst_id = burst_ids[0]
        hp_c.save(update_fields=["source_vsm", "source_burst_id"])

        self.stdout.write(self.style.SUCCESS("Created current/future VSM pair"))

        # =====================================================================
        # 10. DSW Results (4 statistical analyses)
        # =====================================================================
        try:
            DSWResult.objects.create(
                id=f"nlp-cap-{uuid.uuid4().hex[:8]}",
                user=user,
                project=proj_a,
                result_type="spc_capability",
                title="Capability Study — Line A Seal Strength",
                data=json.dumps(
                    {
                        "analysis": "Process Capability Study",
                        "summary": "Seal strength capability analysis for Line A. Cpk=1.12 indicates process is marginally capable but not meeting 1.33 target.",
                        "characteristic": "Seal Strength (N)",
                        "lsl": 8.0,
                        "usl": 15.0,
                        "target": 11.5,
                        "mean": 11.82,
                        "std_dev": 0.95,
                        "cpk": 1.12,
                        "ppk": 0.98,
                        "cp": 1.23,
                        "pp": 1.08,
                        "yield_percent": 97.2,
                        "ppm_total": 28000,
                        "sigma_level": 3.4,
                        "sample_size": 150,
                        "normality_test": {
                            "test": "Anderson-Darling",
                            "statistic": 0.42,
                            "p_value": 0.31,
                            "is_normal": True,
                        },
                        "findings": [
                            "Cpk = 1.12 — below 1.33 minimum for capable process",
                            "Ppk = 0.98 — significant gap between Cp and Pp indicates process shift",
                            "97.2% yield with 28,000 ppm nonconforming",
                            "Distribution is normal (A-D p=0.31)",
                        ],
                    }
                ),
            )
            DSWResult.objects.create(
                id=f"nlp-grr-{uuid.uuid4().hex[:8]}",
                user=user,
                project=proj_a,
                result_type="spc_gage_rr",
                title="Gage R&R — Film Thickness Measurement",
                data=json.dumps(
                    {
                        "analysis": "Gage R&R Study (Crossed)",
                        "summary": "Film thickness measurement system evaluation. %GRR=12.4% — acceptable for process monitoring per AIAG MSA guidelines.",
                        "measurement": "Film Thickness (μm)",
                        "operators": 3,
                        "parts": 10,
                        "trials": 3,
                        "grr_percent": 12.4,
                        "repeatability_percent": 8.1,
                        "reproducibility_percent": 9.3,
                        "part_to_part_percent": 87.6,
                        "ndc": 5,
                        "tolerance": 25.0,
                        "findings": [
                            "%GRR = 12.4% of tolerance — acceptable (< 30%)",
                            "Number of distinct categories = 5 (≥ 5 required)",
                            "Repeatability (8.1%) > Reproducibility (9.3%) — slight operator effect",
                            "Measurement system is adequate for DMAIC analysis",
                        ],
                    }
                ),
            )
            DSWResult.objects.create(
                id=f"nlp-ttest-{uuid.uuid4().hex[:8]}",
                user=user,
                project=proj_a,
                result_type="from_intent",
                title="2-Sample t-Test — Before/After Tension Calibration",
                data=json.dumps(
                    {
                        "analysis": "Two-Sample t-Test",
                        "summary": "Statistically significant improvement in seal strength after tension calibration (p=0.003, d=0.82).",
                        "test_type": "Welch's t-test (unequal variances)",
                        "group_1": {"label": "Before Calibration", "n": 50, "mean": 10.8, "std": 1.2},
                        "group_2": {"label": "After Calibration", "n": 50, "mean": 11.9, "std": 0.9},
                        "t_statistic": 3.07,
                        "p_value": 0.003,
                        "effect_size": 0.82,
                        "confidence_interval": [0.38, 1.82],
                        "power": 0.91,
                        "findings": [
                            "Mean seal strength increased from 10.8 N to 11.9 N after calibration",
                            "Statistically significant (p=0.003) with large effect size (d=0.82)",
                            "95% CI for difference: [0.38, 1.82] N",
                            "Study power: 0.91 (adequate)",
                        ],
                    }
                ),
            )
            DSWResult.objects.create(
                id=f"nlp-doe-{uuid.uuid4().hex[:8]}",
                user=user,
                project=proj_a,
                result_type="from_intent",
                title="DOE — Seal Temperature × Pressure × Dwell Time",
                data=json.dumps(
                    {
                        "analysis": "Full Factorial DOE (2³ + center points)",
                        "summary": "Temperature and pressure are significant factors for seal strength. Interaction effect significant. Optimal: 185°C, 45 PSI, 0.8s.",
                        "factors": [
                            {"name": "Temperature", "low": 175, "high": 195, "unit": "°C"},
                            {"name": "Pressure", "low": 35, "high": 55, "unit": "PSI"},
                            {"name": "Dwell Time", "low": 0.6, "high": 1.0, "unit": "s"},
                        ],
                        "response": "Seal Strength (N)",
                        "runs": 36,
                        "center_points": 4,
                        "anova": [
                            {"source": "Temperature", "df": 1, "f_value": 12.8, "p_value": 0.002, "significant": True},
                            {"source": "Pressure", "df": 1, "f_value": 7.2, "p_value": 0.015, "significant": True},
                            {"source": "Dwell Time", "df": 1, "f_value": 0.95, "p_value": 0.34, "significant": False},
                            {"source": "Temp×Pressure", "df": 1, "f_value": 8.6, "p_value": 0.008, "significant": True},
                            {"source": "Temp×Dwell", "df": 1, "f_value": 0.31, "p_value": 0.58, "significant": False},
                            {
                                "source": "Pressure×Dwell",
                                "df": 1,
                                "f_value": 0.18,
                                "p_value": 0.68,
                                "significant": False,
                            },
                        ],
                        "r_squared": 0.87,
                        "adj_r_squared": 0.83,
                        "optimal": {"Temperature": 185, "Pressure": 45, "Dwell Time": 0.8},
                        "predicted_optimal_response": 12.8,
                        "confirmation": {"n": 50, "mean": 12.6, "failures": 0},
                        "findings": [
                            "Temperature (p=0.002) and Pressure (p=0.015) are significant main effects",
                            "Temperature × Pressure interaction is significant (p=0.008)",
                            "Dwell time is not significant in the tested range",
                            "Optimal settings: 185°C, 45 PSI, 0.8s → predicted 12.8 N",
                            "Confirmation run: 0/50 failures at optimal settings",
                        ],
                    }
                ),
            )
            self.stdout.write(self.style.SUCCESS("Created 4 DSW results"))
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"Skipped DSW results (encryption key may not be set): {exc}"))

        # =====================================================================
        # 11. Quality Tools: A3, FMEA, RCA
        # =====================================================================

        # A3 Report
        a3 = A3Report.objects.create(
            owner=user,
            project=proj_a,
            title="Seal Failure Investigation — Line A",
            status=A3Report.Status.IN_PROGRESS,
            background=(
                "Line A seal failure rate has been running at 8.2% since Q4 2025, "
                "4.2pp above the 4.0% target. Annual cost of excess scrap is $420K. "
                "Two major customer accounts have issued formal corrective action requests."
            ),
            current_condition=(
                "• Scrap rate: 8.2% (target 4.0%)\n"
                "• Cpk on seal strength: 1.12 (target >1.33)\n"
                "• First-hour post-changeover scrap: 22% vs 6% steady-state\n"
                "• Film tension CV: 8.3% (spec <5%)\n"
                "• 3 different changeover procedures observed across shifts"
            ),
            goal=(
                "Reduce seal failure scrap from 8.2% to <4.0% within 12 weeks. "
                "Restore Cpk to >1.33. Sustain for 60 days."
            ),
            root_cause=(
                "Primary: Film tension variability (CV 8.3%) causing inconsistent "
                "seal formation. Driven by: (1) unwind tension not re-optimized for "
                "new film supplier's elastic modulus, (2) no real-time tension monitoring.\n\n"
                "Contributing: Changeover procedure inconsistency causing first-hour "
                "scrap spikes. Three different warm-up sequences, no standard work.\n\n"
                "Eliminated: Humidity (p=0.41, not significant in observed range)."
            ),
            countermeasures=(
                "1. Install real-time tension monitoring with SPC alarming (In Progress)\n"
                "2. Optimize tension setpoint via DOE for new film (Planned)\n"
                "3. Standardize changeover SOP with verification checklist (In Progress)\n"
                "4. Train all shifts on new SOP via TWI Job Instruction (Planned)"
            ),
            implementation_plan=(
                "Week 1-2: Tension monitor installation and commissioning (James Wu)\n"
                "Week 2-3: Changeover SOP development and review (Sarah Martinez)\n"
                "Week 3-4: Confirmation DOE at optimized parameters (Lisa Chen)\n"
                "Week 4-6: Operator training all shifts (Maria Gonzalez)\n"
                "Week 6-18: 60-day sustain and control phase"
            ),
            follow_up=(
                "• Weekly scrap rate review in CI stand-up\n"
                "• SPC control charts on seal strength and tension — daily audit\n"
                "• 30-day and 60-day sustain reviews scheduled\n"
                "• Success metric: scrap <4.0% for 60 consecutive days"
            ),
        )

        # FMEA
        fmea = FMEA.objects.create(
            owner=user,
            project=proj_a,
            title="Packaging Process FMEA — Line A",
            description="Process FMEA for Line A flexible packaging. Covers sealing, slitting, printing, and laminating failure modes.",
            status=FMEA.Status.ACTIVE,
            fmea_type=FMEA.FMEAType.PROCESS,
        )
        fmea_rows = [
            FMEARow(
                fmea=fmea,
                sort_order=0,
                process_step="Sealing",
                failure_mode="Incomplete seal",
                effect="Package leak → customer complaint → product spoilage",
                severity=8,
                cause="Film tension variability exceeding ±5% CV",
                occurrence=7,
                current_controls="Visual inspection (100% but subjective)",
                detection=6,
                recommended_action="Install real-time tension monitoring with SPC",
                action_owner="James Wu",
                action_status=FMEARow.ActionStatus.IN_PROGRESS,
                revised_severity=8,
                revised_occurrence=3,
                revised_detection=3,
                hypothesis_link=h1,
            ),
            FMEARow(
                fmea=fmea,
                sort_order=1,
                process_step="Sealing",
                failure_mode="Weak seal (passes visual, fails pull test)",
                effect="Field failure → recall risk → brand damage",
                severity=9,
                cause="Seal temperature/pressure outside optimal window",
                occurrence=5,
                current_controls="Pull test sampling (1/500 packages)",
                detection=5,
                recommended_action="Implement DOE-optimized parameters and SPC on seal strength",
                action_owner="Lisa Chen",
                action_status=FMEARow.ActionStatus.NOT_STARTED,
            ),
            FMEARow(
                fmea=fmea,
                sort_order=2,
                process_step="Changeover",
                failure_mode="Incorrect heat bar warm-up sequence",
                effect="22% first-hour scrap vs 6% steady-state",
                severity=6,
                cause="No standardized SOP — tribal knowledge",
                occurrence=8,
                current_controls="None — operator discretion",
                detection=8,
                recommended_action="Standardize SOP with verification checklist",
                action_owner="Sarah Martinez",
                action_status=FMEARow.ActionStatus.IN_PROGRESS,
                revised_severity=6,
                revised_occurrence=2,
                revised_detection=3,
                hypothesis_link=h3,
            ),
            FMEARow(
                fmea=fmea,
                sort_order=3,
                process_step="Slitting",
                failure_mode="Edge trim exceeds 3mm specification",
                effect="Material waste increase → cost impact",
                severity=4,
                cause="Blade wear beyond service interval",
                occurrence=3,
                current_controls="Scheduled blade replacement every 50K meters",
                detection=4,
            ),
            FMEARow(
                fmea=fmea,
                sort_order=4,
                process_step="Printing",
                failure_mode="Color density out of spec",
                effect="Customer rejects → rework or scrap entire run",
                severity=7,
                cause="Ink viscosity drift during long runs",
                occurrence=4,
                current_controls="Hourly densitometer check",
                detection=3,
            ),
            FMEARow(
                fmea=fmea,
                sort_order=5,
                process_step="Laminating",
                failure_mode="Delamination under stress",
                effect="Product damage in transit → customer complaint",
                severity=7,
                cause="Adhesive application weight below minimum",
                occurrence=3,
                current_controls="Coat weight check every 2 hours",
                detection=4,
            ),
        ]
        FMEARow.objects.bulk_create(fmea_rows)
        self.stdout.write(self.style.SUCCESS("Created FMEA with 6 failure modes"))

        # RCA Session
        rca = RCASession.objects.create(
            owner=user,
            project=proj_a,
            a3_report=a3,
            title="Customer Complaint — Delaminated Labels Batch 2026-0847",
            event="Customer (National Distributors) reported 2,400 packages from batch 2026-0847 with delaminated labels. Labels separated from packaging during warehouse storage (no unusual conditions). Batch was produced on Feb 3, 2026, Line A, B-shift.",
            chain=[
                {
                    "claim": "Labels delaminated during normal warehouse storage",
                    "critique": "Accepted — customer provided photographic evidence and storage condition logs (68°F, 45% RH).",
                    "accepted": True,
                    "error_labels": [],
                },
                {
                    "claim": "Adhesive bond was insufficient at the time of packaging",
                    "critique": "Accepted — retained samples from same batch confirmed peel strength 40% below spec.",
                    "accepted": True,
                    "error_labels": [],
                },
                {
                    "claim": "Laminator adhesive application was below minimum coat weight",
                    "critique": "Accepted — coat weight log for Feb 3 shows dip to 1.8 g/m² at 14:30 (spec: 2.5 g/m² min).",
                    "accepted": True,
                    "error_labels": [],
                },
                {
                    "claim": "Adhesive supply pump developed intermittent air lock",
                    "critique": "Accepted — maintenance log shows pump was serviced for air lock on Feb 5 after operator complaint.",
                    "accepted": True,
                    "error_labels": [],
                },
                {
                    "claim": "Pump inlet filter was partially clogged, causing cavitation",
                    "critique": "Accepted — filter inspection on Feb 5 found 60% blockage. PM schedule calls for monthly replacement but last change was 47 days prior.",
                    "accepted": True,
                    "error_labels": [],
                },
            ],
            root_cause="Adhesive supply pump inlet filter was 60% clogged (47 days since last replacement vs 30-day PM schedule), causing intermittent cavitation and reduced coat weight. The maintenance PM compliance system failed to flag the overdue filter change.",
            countermeasure="1. Immediate: Replace filter and add real-time coat weight monitoring with low-limit alarm. 2. Systemic: Add filter replacement to CMMS with hard stop escalation at 35 days. 3. Verify: Run 100-piece confirmation batch with coat weight audit every 500 pieces.",
            evaluation="Root cause chain is well-supported with physical evidence at each step. The 5-why chain correctly traces from symptom (delamination) through mechanism (low coat weight) to root cause (clogged filter + PM non-compliance). Countermeasures address both immediate and systemic issues.",
            status=RCASession.Status.COMPLETE,
        )

        # Action item from RCA
        ActionItem.objects.create(
            project=proj_a,
            title="Add coat weight low-limit alarm to laminator SCADA",
            description="Configure real-time alarm when adhesive coat weight drops below 2.2 g/m² (warning) or 2.0 g/m² (stop). Integrate with operator HMI.",
            owner_name="James Wu",
            status=ActionItem.Status.COMPLETED,
            start_date=date.today() - timedelta(days=18),
            end_date=date.today() - timedelta(days=12),
            due_date=date.today() - timedelta(days=14),
            progress=100,
            sort_order=10,
            source_type="rca",
            source_id=rca.id,
        )
        ActionItem.objects.create(
            project=proj_a,
            title="Update CMMS PM schedule with hard-stop escalation for filter replacement",
            description="Set 30-day filter replacement interval with 25-day warning and 35-day hard-stop escalation to maintenance supervisor.",
            owner_name="James Wu",
            status=ActionItem.Status.IN_PROGRESS,
            start_date=date.today() - timedelta(days=10),
            due_date=date.today() + timedelta(days=5),
            progress=70,
            sort_order=11,
            source_type="rca",
            source_id=rca.id,
        )
        self.stdout.write(self.style.SUCCESS("Created A3 report, RCA session, action items"))

        # =====================================================================
        # 12. X-Matrix Correlations
        # =====================================================================
        correlations = [
            # Strategic ↔ Annual
            (so1.id, ao1.id, "strategic_annual", "strong"),
            (so2.id, ao2.id, "strategic_annual", "strong"),
            (so3.id, ao3.id, "strategic_annual", "strong"),
            (so1.id, ao2.id, "strategic_annual", "moderate"),  # OEE also reduces waste
            # Annual ↔ Project
            (ao1.id, hp_a.id, "annual_project", "strong"),
            (ao2.id, hp_b.id, "annual_project", "strong"),
            (ao2.id, hp_c.id, "annual_project", "moderate"),  # SMED supports OEE
            (ao3.id, hp_a.id, "annual_project", "moderate"),  # Scrap project drives SPC
            # Project ↔ KPI
            (hp_a.id, kpi1.id, "project_kpi", "strong"),  # Scrap project → scrap rate
            (hp_a.id, kpi3.id, "project_kpi", "strong"),  # Scrap project → FPY
            (hp_a.id, kpi4.id, "project_kpi", "strong"),  # Scrap project → savings
            (hp_b.id, kpi2.id, "project_kpi", "strong"),  # OEE project → OEE KPI
            (hp_b.id, kpi4.id, "project_kpi", "moderate"),  # OEE project → savings
            (hp_c.id, kpi2.id, "project_kpi", "moderate"),  # SMED → OEE
            # KPI ↔ Strategic
            (kpi1.id, so1.id, "kpi_strategic", "strong"),  # Scrap rate → waste reduction
            (kpi2.id, so2.id, "kpi_strategic", "strong"),  # OEE → world-class OEE
            (kpi3.id, so3.id, "kpi_strategic", "strong"),  # FPY → zero escapes
            (kpi4.id, so1.id, "kpi_strategic", "moderate"),  # Savings → waste reduction
        ]
        for row_id, col_id, pair_type, strength in correlations:
            XMatrixCorrelation.objects.create(
                tenant=tenant,
                fiscal_year=2026,
                pair_type=pair_type,
                row_id=row_id,
                col_id=col_id,
                strength=strength,
                source=XMatrixCorrelation.Source.MANUAL,
                is_confirmed=True,
            )
        self.stdout.write(self.style.SUCCESS(f"Created {len(correlations)} X-Matrix correlations"))

        # =====================================================================
        # Summary
        # =====================================================================
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("NLP Demo Seed Complete"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(f"  Tenant:     {tenant.name} (enterprise)")
        self.stdout.write(f"  User:       {user.username} (id={user.id})")
        self.stdout.write(f"  Site:       {site.name} ({site.code})")
        self.stdout.write("  Strategic:  3 objectives")
        self.stdout.write("  Annual:     3 objectives")
        self.stdout.write("  KPIs:       4")
        self.stdout.write("  Projects:   3 (2 active, 1 completed)")
        self.stdout.write("  Hypotheses: 7 (with Bayesian updates)")
        self.stdout.write("  Evidence:   8 items, linked")
        self.stdout.write("  VSMs:       current/future pair")
        self.stdout.write("  DSW:        4 analyses")
        self.stdout.write("  A3:         1 report")
        self.stdout.write("  FMEA:       1 study (6 rows)")
        self.stdout.write("  RCA:        1 session (5-why)")
        self.stdout.write("  Actions:    7 items")
        self.stdout.write(f"  X-Matrix:   {len(correlations)} correlations")
        self.stdout.write("")

    def _clean(self, user):
        """Remove all NLP demo data."""
        # Tenant cascade will remove Site, StrategicObjective, AnnualObjective,
        # HoshinKPI, XMatrixCorrelation, Membership
        deleted, details = Tenant.objects.filter(slug=NLP_TENANT_SLUG).delete()
        if deleted:
            self.stdout.write(self.style.WARNING(f"Deleted tenant + {deleted} related objects"))

        # Projects (owned by tenant, tagged nlp-demo) cascade to:
        # Hypothesis, Evidence, EvidenceLink, HoshinProject, ActionItem, A3, FMEA, RCA, VSM
        projects = Project.objects.filter(tags__contains=[NLP_TAG])
        if projects.exists():
            count = projects.count()
            projects.delete()
            self.stdout.write(self.style.WARNING(f"Deleted {count} NLP demo projects"))

        # DSW results
        dsw_deleted = DSWResult.objects.filter(user=user, id__startswith="nlp-").delete()[0]
        if dsw_deleted:
            self.stdout.write(self.style.WARNING(f"Deleted {dsw_deleted} DSW results"))

        # VSMs not linked to projects (shouldn't happen, but safety)
        ValueStreamMap.objects.filter(owner=user, name__icontains="Packaging Line A").delete()

        self.stdout.write(self.style.SUCCESS("Clean complete"))

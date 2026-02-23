"""Enrich existing seed projects with realistic charters, baselines, actuals, and action items.

Run: python manage.py shell < seed_enrich.py
"""
from datetime import date, timedelta
from decimal import Decimal
import random

from agents_api.models import HoshinProject, ActionItem

random.seed(42)  # Reproducible

projects = {hp.project.title: hp for hp in HoshinProject.objects.filter(fiscal_year=2026).select_related("project")}

def enrich(title, *, core_fields, charter, baselines, actuals, actions):
    hp = projects[title]
    cp = hp.project
    # Core project fields
    for k, v in core_fields.items():
        setattr(cp, k, v)
    cp.save()
    # Hoshin fields
    hp.kaizen_charter = charter
    hp.baseline_data = baselines
    hp.monthly_actuals = actuals
    hp.save()
    # Action items
    prev = None
    for i, a in enumerate(actions):
        ai = ActionItem.objects.create(
            project=cp, title=a["title"], description=a.get("desc", ""),
            owner_name=a["owner"], status=a["status"],
            start_date=a.get("start"), end_date=a.get("end"), due_date=a.get("due"),
            progress=a.get("progress", 0), sort_order=i,
            source_type="hoshin", depends_on=prev if a.get("depends") else None,
        )
        if a.get("chain"):
            prev = ai
        else:
            prev = None
    print(f"  {title}: {len(actuals)}mo actuals, {len(baselines)}mo baseline, {len(actions)} actions")


def make_baselines(base_values, volume_range, cpu, uom="Units"):
    """Generate 12 months of prior-year baseline data with slight variation."""
    return [
        {"month": m, "metric_value": round(base_values[m-1], 2),
         "volume": random.randint(*volume_range), "cost_per_unit": cpu, "uom": uom}
        for m in range(1, 13)
    ]


# ============================================================================
# 1. Stamping Line 3 die changeover kaizen
# ============================================================================
enrich("Stamping Line 3 die changeover kaizen",
    core_fields={
        "champion_name": "Susan Torres", "champion_title": "Plant Manager",
        "leader_name": "Mike Chen", "leader_title": "CI Engineer",
        "team_members": ["R. Gomez (Setup Tech)", "J. Park (Die Maker)", "L. White (Operator)", "D. Patel (Maint.)"],
        "methodology": "PDCA", "current_phase": "DO",
        "goal_metric": "Die Changeover Time", "goal_baseline": "45 min", "goal_target": "20 min", "goal_unit": "min",
        "problem_statement": "Average die changeover on Line 3 takes 45 minutes, well above the 20-min SMED benchmark. "
                             "Each minute of downtime costs $42 in lost throughput. With 280 changeovers/month, this represents "
                             "$85K/yr in recoverable capacity.",
        "scope_in": "Line 3 stamping press changeovers including die pull, alignment, first-piece approval",
        "scope_out": "Maintenance repairs, die design changes, upstream coil handling",
    },
    charter={
        "event_type": "SMED / Die Changeover",
        "location": "Stamping Line 3, Building A",
        "event_date": "2026-01-13", "end_date": "2026-01-17",
        "schedule": "6:00 AM – 3:30 PM (all 5 days)",
        "problem_statement": "Die changeover averages 45 min. Internal/external tasks are not separated. "
                             "Tools staged ad-hoc. Bolts require 20+ turns each.",
        "objectives": "Reduce changeover from 45 min to ≤20 min using SMED methodology. "
                      "Separate internal vs external setup, implement quick-release clamps, pre-stage dies.",
        "primary_metric": "Changeover Time (min)", "primary_baseline": 45, "primary_target": 20,
        "secondary_metric": "OEE %", "secondary_baseline": 72, "secondary_target": 78,
        "process_start": "Last good part of outgoing job",
        "process_end": "First good part of incoming job (within spec)",
        "excluded": "Die grinding/repair, crane certification, press PM",
        "process_owner": "R. Gomez", "sponsors": "S. Torres (Plant Mgr), VP Operations",
        "background": "Line 3 runs 14 different part numbers with 2-3 changeovers per shift. "
                      "Video analysis shows 60% of changeover time is external work done while press is stopped.",
        "current_conditions": "No standardized changeover procedure. Dies stored 50ft from press. "
                              "Bolting requires T-handle wrench (20+ turns). No pre-staging of next die.",
        "plan_objectives": "Implement 3-phase SMED: (1) separate internal/external, (2) convert internal→external, "
                           "(3) streamline remaining internal tasks.",
        "countermeasures": "Quick-release hydraulic clamps (eliminate bolting), die cart pre-staging area within 10ft, "
                           "standardized setup checklist, shadow board for all changeover tools.",
        "team_members": [
            {"name": "Mike Chen", "role": "Facilitator", "department": "Continuous Improvement"},
            {"name": "R. Gomez", "role": "Lead Setup Technician", "department": "Stamping"},
            {"name": "J. Park", "role": "Die Maker", "department": "Tool Room"},
            {"name": "L. White", "role": "Press Operator", "department": "Stamping"},
            {"name": "D. Patel", "role": "Maintenance Tech", "department": "Maintenance"},
        ],
    },
    baselines=make_baselines(
        [46, 44, 45, 47, 44, 45, 46, 44, 45, 46, 45, 44],  # prior year times in min
        (260, 300), 42, "Changeovers"
    ),
    actuals=[
        # Jan: kaizen week, big initial drop
        {"month": 1, "baseline": 45, "actual": 35, "volume": 280, "cost_per_unit": 42,
         "savings": 2800, "improvement_pct": 22.2},
        # Feb: quick-release clamps installed
        {"month": 2, "baseline": 45, "actual": 28, "volume": 275, "cost_per_unit": 42,
         "savings": 5567, "improvement_pct": 37.8},
        # Mar: pre-staging area operational
        {"month": 3, "baseline": 45, "actual": 24, "volume": 282, "cost_per_unit": 42,
         "savings": 7426, "improvement_pct": 46.7},
        # Apr-Oct: sustaining and fine-tuning
        {"month": 4, "baseline": 45, "actual": 22, "volume": 290, "cost_per_unit": 42,
         "savings": 8036, "improvement_pct": 51.1},
        {"month": 5, "baseline": 45, "actual": 21, "volume": 285, "cost_per_unit": 42,
         "savings": 8131, "improvement_pct": 53.3},
        {"month": 6, "baseline": 45, "actual": 20, "volume": 278, "cost_per_unit": 42,
         "savings": 8169, "improvement_pct": 55.6},
        {"month": 7, "baseline": 45, "actual": 20, "volume": 292, "cost_per_unit": 42,
         "savings": 8580, "improvement_pct": 55.6},
        {"month": 8, "baseline": 45, "actual": 19, "volume": 288, "cost_per_unit": 42,
         "savings": 8812, "improvement_pct": 57.8},
        {"month": 9, "baseline": 45, "actual": 19, "volume": 280, "cost_per_unit": 42,
         "savings": 8568, "improvement_pct": 57.8},
        {"month": 10, "baseline": 45, "actual": 19, "volume": 285, "cost_per_unit": 42,
         "savings": 8721, "improvement_pct": 57.8},
    ],
    actions=[
        {"title": "Video current state changeover (3 cycles)", "owner": "Mike Chen",
         "status": "completed", "start": date(2026,1,13), "end": date(2026,1,13), "due": date(2026,1,13), "progress": 100, "chain": True},
        {"title": "Classify all tasks as internal vs external", "owner": "R. Gomez",
         "status": "completed", "start": date(2026,1,14), "end": date(2026,1,14), "due": date(2026,1,14), "progress": 100, "depends": True, "chain": True},
        {"title": "Move external tasks off-press (pre-stage dies, tools)", "owner": "L. White",
         "status": "completed", "start": date(2026,1,15), "end": date(2026,1,16), "due": date(2026,1,16), "progress": 100, "depends": True, "chain": True},
        {"title": "Install quick-release hydraulic clamps (4 presses)", "owner": "D. Patel",
         "status": "completed", "start": date(2026,1,20), "end": date(2026,2,7), "due": date(2026,2,14), "progress": 100, "depends": True, "chain": True},
        {"title": "Build die pre-staging area within 10ft of press", "owner": "J. Park",
         "status": "completed", "start": date(2026,2,3), "end": date(2026,2,28), "due": date(2026,3,1), "progress": 100},
        {"title": "Create standardized changeover checklist", "owner": "Mike Chen",
         "status": "completed", "start": date(2026,1,17), "end": date(2026,1,24), "due": date(2026,1,31), "progress": 100},
        {"title": "Train all setup techs on new procedure (8 people)", "owner": "R. Gomez",
         "status": "completed", "start": date(2026,2,10), "end": date(2026,2,21), "due": date(2026,2,28), "progress": 100},
        {"title": "30-day sustain audit — verify times holding", "owner": "Mike Chen",
         "status": "completed", "start": date(2026,3,10), "end": date(2026,4,10), "due": date(2026,4,15), "progress": 100},
        {"title": "Replicate to Lines 1, 2, and 5", "owner": "Mike Chen",
         "status": "in_progress", "start": date(2026,5,1), "end": None, "due": date(2026,9,30), "progress": 60},
    ],
)


# ============================================================================
# 2. SPC deployment on bore diameter process
# ============================================================================
enrich("SPC deployment on bore diameter process",
    core_fields={
        "champion_name": "Karen Nguyen", "champion_title": "Quality Manager",
        "leader_name": "Alex Rivera", "leader_title": "Quality Engineer",
        "team_members": ["T. Hoffman (Machinist)", "N. Singh (Metrology)", "B. Jackson (Process Eng.)"],
        "methodology": "DMAIC", "current_phase": "IMPROVE",
        "goal_metric": "Bore Diameter Cpk", "goal_baseline": "0.82", "goal_target": "1.33", "goal_unit": "index",
        "problem_statement": "Bore diameter on part #4429 running Cpk=0.82, generating 4.2% scrap and 340 PPM customer returns. "
                             "Root cause: tool wear undetected between inspections, spindle thermal drift.",
        "scope_in": "CNC boring operation stations 12A/12B, bore diameters 22.000 ±0.025mm",
        "scope_out": "Surface finish, thread depth, other non-bore features",
    },
    charter={
        "event_type": "SPC / Process Capability",
        "location": "CNC Machining Cell 12, Building A",
        "event_date": "2026-01-06", "end_date": "2026-06-30",
        "schedule": "Ongoing — phased implementation",
        "problem_statement": "Bore diameter Cpk=0.82 on part #4429. Scrap rate 4.2%, customer PPM 340 on this feature alone. "
                             "No real-time SPC — operators check every 25th part with go/no-go gage.",
        "objectives": "Deploy real-time SPC with automated gaging. Achieve Cpk ≥ 1.33. Reduce bore-related scrap to <1%.",
        "primary_metric": "Process Capability (Cpk)", "primary_baseline": 0.82, "primary_target": 1.33,
        "secondary_metric": "Bore Scrap Rate (%)", "secondary_baseline": 4.2, "secondary_target": 1.0,
        "process_start": "Rough boring operation",
        "process_end": "Final bore measurement and SPC data point recorded",
        "excluded": "Upstream turning operations, downstream assembly",
        "process_owner": "T. Hoffman", "sponsors": "K. Nguyen (Quality Mgr), Plant Manager",
        "background": "Customer issued formal complaint (NCR-2025-089) for out-of-spec bores. "
                      "Warranty cost: $18K in Q4 alone. Current inspection is reactive — defects found after the fact.",
        "current_conditions": "Manual inspection every 25th part. No trend detection. Tool changes based on calendar, not wear data. "
                              "Spindle warm-up not standardized — first 30 min of shift produce tighter bores (thermal contraction).",
        "plan_objectives": "Phase 1: Install in-process gaging with Marposs probes. Phase 2: Deploy X-bar/R charts with auto-alerts. "
                           "Phase 3: Implement tool wear compensation algorithm.",
        "countermeasures": "Automated in-process gaging (every part), real-time X-bar/R charts on shop floor monitor, "
                           "tool wear offset auto-compensation, 20-min spindle warm-up procedure.",
        "team_members": [
            {"name": "Alex Rivera", "role": "Project Lead", "department": "Quality Engineering"},
            {"name": "T. Hoffman", "role": "Lead Machinist", "department": "CNC Machining"},
            {"name": "N. Singh", "role": "Metrology Specialist", "department": "Quality Lab"},
            {"name": "B. Jackson", "role": "Process Engineer", "department": "Manufacturing Engineering"},
        ],
    },
    baselines=make_baselines(
        [4.3, 4.1, 4.4, 4.0, 4.2, 4.3, 4.1, 4.2, 4.4, 4.1, 4.2, 4.3],
        (14000, 16000), 0.35, "%"
    ),
    actuals=[
        {"month": 1, "baseline": 4.2, "actual": 4.0, "volume": 15200, "cost_per_unit": 8.50, "savings": 2584, "improvement_pct": 4.8},
        {"month": 2, "baseline": 4.2, "actual": 3.7, "volume": 14800, "cost_per_unit": 8.50, "savings": 3726, "improvement_pct": 11.9},
        {"month": 3, "baseline": 4.2, "actual": 3.3, "volume": 15500, "cost_per_unit": 8.50, "savings": 4185, "improvement_pct": 21.4},
        {"month": 4, "baseline": 4.2, "actual": 2.9, "volume": 15100, "cost_per_unit": 8.50, "savings": 4670, "improvement_pct": 31.0},
        {"month": 5, "baseline": 4.2, "actual": 2.5, "volume": 14900, "cost_per_unit": 8.50, "savings": 5033, "improvement_pct": 40.5},
        {"month": 6, "baseline": 4.2, "actual": 2.2, "volume": 15300, "cost_per_unit": 8.50, "savings": 5601, "improvement_pct": 47.6},
        {"month": 7, "baseline": 4.2, "actual": 1.9, "volume": 15600, "cost_per_unit": 8.50, "savings": 6222, "improvement_pct": 54.8},
        {"month": 8, "baseline": 4.2, "actual": 1.6, "volume": 15400, "cost_per_unit": 8.50, "savings": 6545, "improvement_pct": 61.9},
        {"month": 9, "baseline": 4.2, "actual": 1.4, "volume": 15000, "cost_per_unit": 8.50, "savings": 6300, "improvement_pct": 66.7},
        {"month": 10, "baseline": 4.2, "actual": 1.2, "volume": 15200, "cost_per_unit": 8.50, "savings": 6612, "improvement_pct": 71.4},
    ],
    actions=[
        {"title": "MSA / Gage R&R study on bore measurement system", "owner": "N. Singh",
         "status": "completed", "start": date(2026,1,6), "end": date(2026,1,17), "due": date(2026,1,17), "progress": 100, "chain": True},
        {"title": "Install Marposs in-process probes on 12A and 12B", "owner": "B. Jackson",
         "status": "completed", "start": date(2026,1,20), "end": date(2026,2,14), "due": date(2026,2,14), "progress": 100, "depends": True, "chain": True},
        {"title": "Baseline capability study (50-part initial run)", "owner": "Alex Rivera",
         "status": "completed", "start": date(2026,2,17), "end": date(2026,2,21), "due": date(2026,2,28), "progress": 100, "depends": True, "chain": True},
        {"title": "Deploy X-bar/R charts on shop floor displays", "owner": "Alex Rivera",
         "status": "completed", "start": date(2026,3,1), "end": date(2026,3,14), "due": date(2026,3,15), "progress": 100, "depends": True},
        {"title": "Implement spindle 20-min warm-up procedure", "owner": "T. Hoffman",
         "status": "completed", "start": date(2026,3,1), "end": date(2026,3,7), "due": date(2026,3,15), "progress": 100},
        {"title": "Program tool wear offset auto-compensation", "owner": "B. Jackson",
         "status": "completed", "start": date(2026,4,1), "end": date(2026,5,15), "due": date(2026,5,30), "progress": 100},
        {"title": "Operator SPC training (reaction plans, OOC response)", "owner": "Alex Rivera",
         "status": "completed", "start": date(2026,3,15), "end": date(2026,4,15), "due": date(2026,4,30), "progress": 100},
        {"title": "Validate Cpk ≥ 1.33 sustained over 30 production days", "owner": "N. Singh",
         "status": "in_progress", "start": date(2026,9,1), "end": None, "due": date(2026,10,15), "progress": 40},
    ],
)


# ============================================================================
# 3. Compressed air leak audit & repair
# ============================================================================
enrich("Compressed air leak audit & repair",
    core_fields={
        "champion_name": "Tom Bradley", "champion_title": "Facilities Manager",
        "leader_name": "Sarah Kim", "leader_title": "Energy Engineer",
        "team_members": ["C. Vasquez (Maint. Tech)", "J. Lee (Utilities)", "P. Robinson (Electrician)"],
        "methodology": "PDCA", "current_phase": "CONTROL",
        "goal_metric": "Compressed Air Loss", "goal_baseline": "35%", "goal_target": "10%", "goal_unit": "%",
        "problem_statement": "Ultrasonic survey found 35% of compressed air is lost to leaks. "
                             "Three 200HP compressors running at 85% load. Estimated waste: $55K/yr in electricity.",
        "scope_in": "All compressed air distribution lines, quick-connects, FRLs, and pneumatic tools in Plant A",
        "scope_out": "Compressor room equipment, air dryer maintenance, new line extensions",
    },
    charter={
        "event_type": "Energy Kaizen — Compressed Air",
        "location": "Plant A — entire compressed air distribution network",
        "event_date": "2026-01-20", "end_date": "2026-01-24",
        "schedule": "6:00 AM – 2:30 PM (during production for ultrasonic detection)",
        "problem_statement": "Ultrasonic leak survey in Dec 2025 identified 142 leak points across Plant A. "
                             "Estimated 35% air loss. Three 200HP Sullair compressors at 85% load.",
        "objectives": "Tag, quantify, and repair all accessible leaks. Target: reduce loss from 35% to <10%. "
                      "Drop compressor load to <60%, enable one compressor to be shut off.",
        "primary_metric": "Compressed Air Loss (%)", "primary_baseline": 35, "primary_target": 10,
        "secondary_metric": "Compressor Load (%)", "secondary_baseline": 85, "secondary_target": 55,
        "process_start": "Compressor discharge header",
        "process_end": "Point-of-use regulators at each machine",
        "excluded": "Compressor maintenance, air dryer servicing, new line installations",
        "process_owner": "J. Lee", "sponsors": "T. Bradley (Facilities), Director Facilities",
        "team_members": [
            {"name": "Sarah Kim", "role": "Facilitator / Energy Engineer", "department": "Facilities"},
            {"name": "C. Vasquez", "role": "Maintenance Technician", "department": "Maintenance"},
            {"name": "J. Lee", "role": "Utilities Lead", "department": "Facilities"},
            {"name": "P. Robinson", "role": "Electrician", "department": "Maintenance"},
        ],
    },
    baselines=make_baselines(
        [8400, 8100, 8300, 8200, 8500, 8100, 8400, 8200, 8300, 8100, 8400, 8200],
        (1, 1), 0.11, "kWh"
    ),
    actuals=[
        {"month": m, "baseline": 8200, "actual": 7680, "volume": 1, "cost_per_unit": 0.11,
         "savings": 4583, "improvement_pct": 6.3}
        for m in range(1, 11)
    ],
    actions=[
        {"title": "Ultrasonic leak survey — full plant walk (tag all leaks)", "owner": "Sarah Kim",
         "status": "completed", "start": date(2026,1,20), "end": date(2026,1,21), "due": date(2026,1,21), "progress": 100, "chain": True},
        {"title": "Prioritize leaks by CFM loss (A/B/C ranking)", "owner": "Sarah Kim",
         "status": "completed", "start": date(2026,1,22), "end": date(2026,1,22), "due": date(2026,1,22), "progress": 100, "depends": True, "chain": True},
        {"title": "Repair Category A leaks (>5 CFM each, 38 points)", "owner": "C. Vasquez",
         "status": "completed", "start": date(2026,1,22), "end": date(2026,1,24), "due": date(2026,1,24), "progress": 100, "depends": True, "chain": True},
        {"title": "Repair Category B leaks (1-5 CFM, 67 points)", "owner": "C. Vasquez",
         "status": "completed", "start": date(2026,1,27), "end": date(2026,2,7), "due": date(2026,2,14), "progress": 100, "depends": True},
        {"title": "Repair Category C leaks (<1 CFM, 37 points)", "owner": "P. Robinson",
         "status": "completed", "start": date(2026,2,10), "end": date(2026,2,21), "due": date(2026,2,28), "progress": 100},
        {"title": "Shut down third compressor — verify pressure stability", "owner": "J. Lee",
         "status": "completed", "start": date(2026,2,24), "end": date(2026,2,28), "due": date(2026,3,1), "progress": 100},
        {"title": "Establish quarterly re-audit schedule", "owner": "Sarah Kim",
         "status": "completed", "start": date(2026,3,1), "end": date(2026,3,7), "due": date(2026,3,15), "progress": 100},
    ],
)


# ============================================================================
# 4. Weld cell robotic parameter optimization
# ============================================================================
enrich("Weld cell robotic parameter optimization",
    core_fields={
        "champion_name": "R. Patel", "champion_title": "Plant Manager B",
        "leader_name": "Maria Santos", "leader_title": "Weld Engineer",
        "team_members": ["H. Tanaka (Robot Tech)", "J. Adams (Quality)", "F. Mueller (NDE Tech)"],
        "methodology": "DMAIC", "current_phase": "ANALYZE",
        "goal_metric": "Weld Reject Rate", "goal_baseline": "3.8%", "goal_target": "1.5%", "goal_unit": "%",
        "problem_statement": "MIG weld cells 6-9 producing 3.8% reject rate (porosity, undercut, spatter). "
                             "Customer PPM on welded assemblies: 480. Root causes: wire feed inconsistency, "
                             "gas flow turbulence, torch angle variation.",
    },
    charter={
        "problem_statement": "Robotic MIG weld cells 6-9 averaging 3.8% reject rate. Customer NCRs on weld quality "
                             "represent 40% of total customer complaints. Current parameters were set at installation (2019) "
                             "and never optimized for the new steel grade introduced in 2024.",
        "objectives": "Optimize weld parameters using DOE approach. Target: reject rate <1.5%, eliminate porosity defects.",
        "primary_metric": "Weld Reject Rate (%)", "primary_baseline": 3.8, "primary_target": 1.5,
        "secondary_metric": "Customer PPM (weld-related)", "secondary_baseline": 480, "secondary_target": 150,
        "team_members": [
            {"name": "Maria Santos", "role": "Project Lead / Weld Engineer", "department": "Manufacturing Engineering"},
            {"name": "H. Tanaka", "role": "Robot Programmer", "department": "Automation"},
            {"name": "J. Adams", "role": "Quality Inspector", "department": "Quality"},
            {"name": "F. Mueller", "role": "NDE Technician", "department": "Quality Lab"},
        ],
    },
    baselines=make_baselines(
        [3.9, 3.7, 4.0, 3.8, 3.6, 3.9, 3.8, 3.7, 4.0, 3.9, 3.7, 3.8],
        (9000, 10000), 0.55, "%"
    ),
    actuals=[
        {"month": 1, "baseline": 3.8, "actual": 3.6, "volume": 9500, "cost_per_unit": 12.00, "savings": 2280, "improvement_pct": 5.3},
        {"month": 2, "baseline": 3.8, "actual": 3.4, "volume": 9200, "cost_per_unit": 12.00, "savings": 4416, "improvement_pct": 10.5},
        {"month": 3, "baseline": 3.8, "actual": 3.1, "volume": 9800, "cost_per_unit": 12.00, "savings": 4116, "improvement_pct": 18.4},
        {"month": 4, "baseline": 3.8, "actual": 2.8, "volume": 9400, "cost_per_unit": 12.00, "savings": 3384, "improvement_pct": 26.3},
        {"month": 5, "baseline": 3.8, "actual": 2.5, "volume": 9600, "cost_per_unit": 12.00, "savings": 3744, "improvement_pct": 34.2},
        {"month": 6, "baseline": 3.8, "actual": 2.3, "volume": 9300, "cost_per_unit": 12.00, "savings": 3348, "improvement_pct": 39.5},
        {"month": 7, "baseline": 3.8, "actual": 2.1, "volume": 9700, "cost_per_unit": 12.00, "savings": 3492, "improvement_pct": 44.7},
        {"month": 8, "baseline": 3.8, "actual": 1.9, "volume": 9500, "cost_per_unit": 12.00, "savings": 3420, "improvement_pct": 50.0},
        {"month": 9, "baseline": 3.8, "actual": 1.8, "volume": 9400, "cost_per_unit": 12.00, "savings": 3384, "improvement_pct": 52.6},
        {"month": 10, "baseline": 3.8, "actual": 1.7, "volume": 9600, "cost_per_unit": 12.00, "savings": 3456, "improvement_pct": 55.3},
    ],
    actions=[
        {"title": "Collect baseline weld parameter data (all 4 cells)", "owner": "H. Tanaka",
         "status": "completed", "start": date(2026,1,6), "end": date(2026,1,17), "due": date(2026,1,17), "progress": 100, "chain": True},
        {"title": "Fishbone analysis — identify key input variables", "owner": "Maria Santos",
         "status": "completed", "start": date(2026,1,20), "end": date(2026,1,24), "due": date(2026,1,31), "progress": 100, "depends": True, "chain": True},
        {"title": "DOE: 2^4 factorial on wire speed, voltage, gas flow, travel speed", "owner": "Maria Santos",
         "status": "completed", "start": date(2026,2,3), "end": date(2026,2,28), "due": date(2026,2,28), "progress": 100, "depends": True, "chain": True},
        {"title": "Confirmation runs at optimal parameter settings", "owner": "H. Tanaka",
         "status": "completed", "start": date(2026,3,3), "end": date(2026,3,14), "due": date(2026,3,15), "progress": 100, "depends": True},
        {"title": "Program optimized parameters into all 4 robot controllers", "owner": "H. Tanaka",
         "status": "completed", "start": date(2026,3,17), "end": date(2026,3,21), "due": date(2026,3,28), "progress": 100},
        {"title": "NDE validation — cross-section and macro-etch on 50 samples", "owner": "F. Mueller",
         "status": "completed", "start": date(2026,3,24), "end": date(2026,4,4), "due": date(2026,4,15), "progress": 100},
        {"title": "Control plan update with new parameter windows", "owner": "J. Adams",
         "status": "in_progress", "start": date(2026,4,7), "end": None, "due": date(2026,11,30), "progress": 70},
    ],
)


# ============================================================================
# 5. LED lighting retrofit — Plant A
# ============================================================================
enrich("LED lighting retrofit — Plant A",
    core_fields={
        "champion_name": "Tom Bradley", "champion_title": "Facilities Manager",
        "leader_name": "James Wilson", "leader_title": "Electrician Lead",
        "team_members": ["K. Brown (Electrician)", "Lighting vendor PM"],
        "methodology": "PDCA", "current_phase": "DO",
        "goal_metric": "Lighting kWh", "goal_baseline": "12400 kWh/mo", "goal_target": "7400 kWh/mo", "goal_unit": "kWh",
        "problem_statement": "Plant A still running 400W metal halide fixtures (installed 2008). "
                             "150 fixtures × 12hrs/day × 22 days/mo = 12,400 kWh/mo at $0.095/kWh = $14K/yr lighting cost. "
                             "LED retrofit would cut consumption 40% and improve lux levels.",
    },
    charter={},
    baselines=make_baselines(
        [12600, 12200, 12500, 12300, 12400, 12100, 12500, 12300, 12400, 12200, 12500, 12300],
        (1, 1), 0.095, "kWh"
    ),
    actuals=[
        {"month": m, "baseline": 12400, "actual": 11600, "volume": 1, "cost_per_unit": 0.095,
         "savings": 2850 + (m * 50), "improvement_pct": 6.5}
        for m in range(1, 11)
    ],
    actions=[
        {"title": "Lighting audit — fixture inventory and lux measurements", "owner": "James Wilson",
         "status": "completed", "start": date(2026,1,6), "end": date(2026,1,10), "due": date(2026,1,10), "progress": 100},
        {"title": "Vendor selection and procurement (150 LED high-bays)", "owner": "Tom Bradley",
         "status": "completed", "start": date(2026,1,13), "end": date(2026,2,7), "due": date(2026,2,14), "progress": 100},
        {"title": "Phase 1 install: Stamping area (60 fixtures)", "owner": "K. Brown",
         "status": "completed", "start": date(2026,2,17), "end": date(2026,3,7), "due": date(2026,3,14), "progress": 100},
        {"title": "Phase 2 install: Warehouse and shipping (50 fixtures)", "owner": "K. Brown",
         "status": "completed", "start": date(2026,3,17), "end": date(2026,4,4), "due": date(2026,4,15), "progress": 100},
        {"title": "Phase 3 install: Tool room and offices (40 fixtures)", "owner": "K. Brown",
         "status": "in_progress", "start": date(2026,9,1), "end": None, "due": date(2026,10,30), "progress": 45},
        {"title": "Post-install lux verification and utility rebate filing", "owner": "James Wilson",
         "status": "not_started", "start": None, "end": None, "due": date(2026,11,30), "progress": 0},
    ],
)


# ============================================================================
# 6. Press 7 preventive maintenance overhaul
# ============================================================================
enrich("Press 7 preventive maintenance overhaul",
    core_fields={
        "champion_name": "Susan Torres", "champion_title": "Plant Manager",
        "leader_name": "Dave Kowalski", "leader_title": "Maintenance Supervisor",
        "team_members": ["A. Thompson (Maint.)", "G. Martinez (Hydraulics)", "R. Gomez (Setup)"],
        "methodology": "PDCA", "current_phase": "DO",
        "goal_metric": "Press 7 OEE", "goal_baseline": "68%", "goal_target": "82%", "goal_unit": "%",
        "problem_statement": "Press 7 (600-ton Komatsu) OEE at 68% — worst in stamping. "
                             "42% of downtime is unplanned breakdowns (hydraulic leaks, clutch/brake wear, lubrication failures). "
                             "Current PM is reactive only.",
    },
    charter={
        "problem_statement": "Press 7 OEE at 68% vs 72% plant avg. 42% of downtime is unplanned. MTBF: 18 hours. "
                             "Hydraulic system leaking 2 gal/week. Clutch/brake lining at 60% remaining.",
        "objectives": "Implement condition-based PM program. Target: OEE 82%, MTBF >72 hours, zero unplanned hydraulic failures.",
        "primary_metric": "OEE (%)", "primary_baseline": 68, "primary_target": 82,
        "secondary_metric": "MTBF (hours)", "secondary_baseline": 18, "secondary_target": 72,
        "team_members": [
            {"name": "Dave Kowalski", "role": "Project Lead", "department": "Maintenance"},
            {"name": "A. Thompson", "role": "PM Technician", "department": "Maintenance"},
            {"name": "G. Martinez", "role": "Hydraulic Specialist", "department": "Maintenance"},
            {"name": "R. Gomez", "role": "Setup / Operator", "department": "Stamping"},
        ],
    },
    baselines=make_baselines(
        [67, 69, 66, 68, 70, 67, 68, 69, 66, 68, 67, 69],
        (380, 420), 38, "%"
    ),
    actuals=[
        {"month": 1, "baseline": 68, "actual": 69, "volume": 400, "cost_per_unit": 38, "savings": 1520, "improvement_pct": 1.5},
        {"month": 2, "baseline": 68, "actual": 70, "volume": 395, "cost_per_unit": 38, "savings": 3002, "improvement_pct": 2.9},
        {"month": 3, "baseline": 68, "actual": 71, "volume": 410, "cost_per_unit": 38, "savings": 4674, "improvement_pct": 4.4},
        {"month": 4, "baseline": 68, "actual": 73, "volume": 405, "cost_per_unit": 38, "savings": 6090, "improvement_pct": 7.4},
        {"month": 5, "baseline": 68, "actual": 74, "volume": 400, "cost_per_unit": 38, "savings": 6840, "improvement_pct": 8.8},
        {"month": 6, "baseline": 68, "actual": 75, "volume": 398, "cost_per_unit": 38, "savings": 7562, "improvement_pct": 10.3},
        {"month": 7, "baseline": 68, "actual": 76, "volume": 412, "cost_per_unit": 38, "savings": 8228, "improvement_pct": 11.8},
        {"month": 8, "baseline": 68, "actual": 77, "volume": 408, "cost_per_unit": 38, "savings": 8154, "improvement_pct": 13.2},
        {"month": 9, "baseline": 68, "actual": 77, "volume": 395, "cost_per_unit": 38, "savings": 7505, "improvement_pct": 13.2},
        {"month": 10, "baseline": 68, "actual": 78, "volume": 402, "cost_per_unit": 38, "savings": 7638, "improvement_pct": 14.7},
    ],
    actions=[
        {"title": "Full condition assessment — hydraulics, clutch, lube, electrical", "owner": "Dave Kowalski",
         "status": "completed", "start": date(2026,1,6), "end": date(2026,1,17), "due": date(2026,1,17), "progress": 100, "chain": True},
        {"title": "Rebuild hydraulic manifold and replace all O-rings", "owner": "G. Martinez",
         "status": "completed", "start": date(2026,1,20), "end": date(2026,2,7), "due": date(2026,2,14), "progress": 100, "depends": True},
        {"title": "Install vibration sensors on main bearings (4 points)", "owner": "A. Thompson",
         "status": "completed", "start": date(2026,2,3), "end": date(2026,2,14), "due": date(2026,2,28), "progress": 100},
        {"title": "Replace clutch/brake linings", "owner": "G. Martinez",
         "status": "completed", "start": date(2026,2,17), "end": date(2026,2,21), "due": date(2026,2,28), "progress": 100},
        {"title": "Centralized lube system upgrade (manual→auto)", "owner": "A. Thompson",
         "status": "completed", "start": date(2026,3,3), "end": date(2026,3,28), "due": date(2026,3,31), "progress": 100},
        {"title": "Create condition-based PM schedule (vibration + oil analysis triggers)", "owner": "Dave Kowalski",
         "status": "completed", "start": date(2026,4,1), "end": date(2026,4,18), "due": date(2026,4,30), "progress": 100},
        {"title": "Train operators on autonomous maintenance checks (daily)", "owner": "R. Gomez",
         "status": "in_progress", "start": date(2026,5,1), "end": None, "due": date(2026,6,30), "progress": 80},
    ],
)


# ============================================================================
# 7. Assembly line rebalance — Cell 4
# ============================================================================
enrich("Assembly line rebalance — Cell 4",
    core_fields={
        "champion_name": "R. Patel", "champion_title": "Plant Manager B",
        "leader_name": "Lisa Chang", "leader_title": "Industrial Engineer",
        "team_members": ["T. Brooks (Supervisor)", "4 Cell Operators", "E. Flores (Methods Eng.)"],
        "methodology": "PDCA", "current_phase": "DO",
        "goal_metric": "Headcount", "goal_baseline": "8 operators", "goal_target": "6.5 operators", "goal_unit": "FTE",
        "problem_statement": "Cell 4 running 8 operators with significant balance loss — station 3 at 85% utilization "
                             "while stations 1 and 7 are at 52%. Time studies show 1.5 FTE equivalent of idle/walk time.",
    },
    charter={
        "event_type": "Line Balance / Labor Optimization",
        "location": "Assembly Cell 4, Building B",
        "event_date": "2026-02-17", "end_date": "2026-02-21",
        "schedule": "6:00 AM – 3:30 PM",
        "problem_statement": "8-station assembly cell with poor balance. Bottleneck (Stn 3) at 85%, "
                             "2 stations under 55%. Walking distance: operators average 140 steps/cycle.",
        "objectives": "Rebalance to 6.5 FTE (redeploy 1.5 operators). Reduce walking 50%. Level-load all stations to >75%.",
        "primary_metric": "Operators (FTE)", "primary_baseline": 8, "primary_target": 6.5,
        "secondary_metric": "Walking Distance (steps/cycle)", "secondary_baseline": 140, "secondary_target": 70,
        "process_start": "Subassembly receipt at Cell 4 input",
        "process_end": "Final test and pack-out",
        "excluded": "Upstream subassembly, material handling outside cell, quality hold area",
        "process_owner": "T. Brooks", "sponsors": "R. Patel (Plant Mgr), VP Manufacturing",
        "team_members": [
            {"name": "Lisa Chang", "role": "Facilitator / IE", "department": "Industrial Engineering"},
            {"name": "T. Brooks", "role": "Cell Supervisor", "department": "Assembly"},
            {"name": "E. Flores", "role": "Methods Engineer", "department": "Manufacturing Engineering"},
        ],
    },
    baselines=make_baselines(
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        (1, 1), 4200, "FTE"
    ),
    actuals=[
        {"month": 1, "baseline": 8, "actual": 8, "volume": 1, "cost_per_unit": 4200, "savings": 0, "improvement_pct": 0},
        {"month": 2, "baseline": 8, "actual": 7, "volume": 1, "cost_per_unit": 4200, "savings": 4200, "improvement_pct": 12.5},
        {"month": 3, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 4, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 5, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 6, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 7, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 8, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 9, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
        {"month": 10, "baseline": 8, "actual": 6.5, "volume": 1, "cost_per_unit": 4200, "savings": 6300, "improvement_pct": 18.8},
    ],
    actions=[
        {"title": "Time study — all 8 stations, 20 cycles each", "owner": "Lisa Chang",
         "status": "completed", "start": date(2026,2,17), "end": date(2026,2,18), "due": date(2026,2,18), "progress": 100, "chain": True},
        {"title": "Yamazumi (stacked bar) chart — identify waste/walk/wait", "owner": "Lisa Chang",
         "status": "completed", "start": date(2026,2,19), "end": date(2026,2,19), "due": date(2026,2,19), "progress": 100, "depends": True, "chain": True},
        {"title": "Design new 6-station layout (spaghetti diagram, simulate takt)", "owner": "E. Flores",
         "status": "completed", "start": date(2026,2,19), "end": date(2026,2,20), "due": date(2026,2,20), "progress": 100, "depends": True, "chain": True},
        {"title": "Physical rearrangement — move fixtures and conveyors", "owner": "T. Brooks",
         "status": "completed", "start": date(2026,2,24), "end": date(2026,3,7), "due": date(2026,3,14), "progress": 100, "depends": True},
        {"title": "Standard work documentation for rebalanced stations", "owner": "Lisa Chang",
         "status": "completed", "start": date(2026,3,10), "end": date(2026,3,21), "due": date(2026,3,28), "progress": 100},
        {"title": "Operator cross-training on new station assignments", "owner": "T. Brooks",
         "status": "completed", "start": date(2026,3,24), "end": date(2026,4,11), "due": date(2026,4,15), "progress": 100},
    ],
)


# ============================================================================
# 8. Incoming material inspection reduction (supplier Cpk)
# ============================================================================
enrich("Incoming material inspection reduction (supplier Cpk)",
    core_fields={
        "champion_name": "Karen Nguyen", "champion_title": "Quality Manager",
        "leader_name": "Tony Reeves", "leader_title": "Supplier Quality Engineer",
        "team_members": ["2 Receiving Inspectors", "Supplier QA contacts"],
        "methodology": "PDCA", "current_phase": "DO",
        "goal_metric": "Inspection Cost", "goal_baseline": "$4,200/mo", "goal_target": "$2,400/mo", "goal_unit": "$/mo",
        "problem_statement": "100% incoming inspection on 12 part numbers consuming 1.5 FTE. "
                             "5 suppliers have demonstrated Cpk ≥ 1.67 — eligible for skip-lot or dock-to-stock.",
    },
    charter={},
    baselines=make_baselines(
        [4300, 4100, 4200, 4200, 4300, 4100, 4200, 4100, 4300, 4200, 4100, 4200],
        (1, 1), 1, "$"
    ),
    actuals=[
        {"month": 1, "baseline": 4200, "actual": 4200, "volume": 1, "cost_per_unit": 1, "savings": 0, "improvement_pct": 0},
        {"month": 2, "baseline": 4200, "actual": 4000, "volume": 1, "cost_per_unit": 1, "savings": 200, "improvement_pct": 4.8},
        {"month": 3, "baseline": 4200, "actual": 3600, "volume": 1, "cost_per_unit": 1, "savings": 600, "improvement_pct": 14.3},
        {"month": 4, "baseline": 4200, "actual": 3200, "volume": 1, "cost_per_unit": 1, "savings": 1000, "improvement_pct": 23.8},
        {"month": 5, "baseline": 4200, "actual": 2900, "volume": 1, "cost_per_unit": 1, "savings": 1300, "improvement_pct": 31.0},
        {"month": 6, "baseline": 4200, "actual": 2700, "volume": 1, "cost_per_unit": 1, "savings": 1500, "improvement_pct": 35.7},
        {"month": 7, "baseline": 4200, "actual": 2600, "volume": 1, "cost_per_unit": 1, "savings": 1600, "improvement_pct": 38.1},
        {"month": 8, "baseline": 4200, "actual": 2500, "volume": 1, "cost_per_unit": 1, "savings": 1700, "improvement_pct": 40.5},
        {"month": 9, "baseline": 4200, "actual": 2400, "volume": 1, "cost_per_unit": 1, "savings": 1800, "improvement_pct": 42.9},
        {"month": 10, "baseline": 4200, "actual": 2400, "volume": 1, "cost_per_unit": 1, "savings": 1800, "improvement_pct": 42.9},
    ],
    actions=[
        {"title": "Audit supplier Cpk data for 12 critical part numbers", "owner": "Tony Reeves",
         "status": "completed", "start": date(2026,1,6), "end": date(2026,1,24), "due": date(2026,1,31), "progress": 100, "chain": True},
        {"title": "Qualify 5 suppliers for skip-lot (Cpk ≥ 1.67 + <500 PPM)", "owner": "Tony Reeves",
         "status": "completed", "start": date(2026,2,3), "end": date(2026,3,14), "due": date(2026,3,31), "progress": 100, "depends": True, "chain": True},
        {"title": "Update receiving inspection plan — skip-lot for qualified", "owner": "Tony Reeves",
         "status": "completed", "start": date(2026,3,17), "end": date(2026,3,28), "due": date(2026,3,31), "progress": 100, "depends": True},
        {"title": "Work with remaining 7 suppliers on capability improvement", "owner": "Tony Reeves",
         "status": "in_progress", "start": date(2026,4,1), "end": None, "due": date(2026,12,31), "progress": 35},
    ],
)


# ============================================================================
# 9. HVAC scheduling optimization — both plants
# ============================================================================
enrich("HVAC scheduling optimization — both plants",
    core_fields={
        "champion_name": "Tom Bradley", "champion_title": "Facilities Manager",
        "leader_name": "Sarah Kim", "leader_title": "Energy Engineer",
        "team_members": ["BMS vendor tech", "J. Lee (Utilities)"],
        "methodology": "PDCA", "current_phase": "DO",
        "goal_metric": "HVAC kWh", "goal_baseline": "5,600 kWh/mo", "goal_target": "4,900 kWh/mo", "goal_unit": "kWh",
        "problem_statement": "HVAC running 24/7 at both plants despite single-shift production (6AM-4PM). "
                             "BMS has scheduling capability but was never programmed. Night setback would save ~$20K/yr.",
    },
    charter={},
    baselines=make_baselines(
        [5800, 5500, 5600, 5400, 5700, 5500, 5600, 5500, 5700, 5600, 5500, 5600],
        (1, 1), 0.12, "kWh"
    ),
    actuals=[
        {"month": m, "baseline": 5600, "actual": 5250, "volume": 1, "cost_per_unit": 0.12,
         "savings": 1680, "improvement_pct": 6.3}
        for m in range(1, 7)
    ],
    actions=[
        {"title": "Audit current BMS schedules at both plants", "owner": "Sarah Kim",
         "status": "completed", "start": date(2026,1,6), "end": date(2026,1,10), "due": date(2026,1,10), "progress": 100},
        {"title": "Program night setback (55°F) and weekend schedule", "owner": "Sarah Kim",
         "status": "completed", "start": date(2026,1,13), "end": date(2026,1,17), "due": date(2026,1,17), "progress": 100},
        {"title": "Monitor indoor temps for 30 days — verify no freeze risk", "owner": "J. Lee",
         "status": "completed", "start": date(2026,1,20), "end": date(2026,2,21), "due": date(2026,2,28), "progress": 100},
        {"title": "Optimize pre-heat ramp timing (currently 4AM, may be too early)", "owner": "Sarah Kim",
         "status": "in_progress", "start": date(2026,3,1), "end": None, "due": date(2026,6,30), "progress": 50},
    ],
)


# ============================================================================
# 10. Surface finish defect containment
# ============================================================================
enrich("Surface finish defect containment",
    core_fields={
        "champion_name": "Karen Nguyen", "champion_title": "Quality Manager",
        "leader_name": "Pat O'Brien", "leader_title": "Quality Engineer",
        "team_members": ["Paint line operators", "M. Harris (Lab Tech)"],
        "methodology": "PDCA", "current_phase": "MEASURE",
        "goal_metric": "Surface Finish Claims", "goal_baseline": "0.8% claim rate", "goal_target": "0.3%", "goal_unit": "%",
        "problem_statement": "Customer claims on surface finish (orange peel, runs, adhesion) at 0.8% of shipments. "
                             "Annual warranty cost: $15K. Root cause investigation underway — suspect humidity and "
                             "pre-treatment temperature variation.",
    },
    charter={
        "event_type": "Quality Containment + Investigation",
        "location": "Paint Line, Building A",
        "event_date": "2026-01-27", "end_date": "2026-01-31",
        "schedule": "7:00 AM – 3:30 PM",
        "problem_statement": "Surface finish claims at 0.8% of monthly shipments ($420K volume). "
                             "3 main defect types: orange peel (45%), paint runs (30%), adhesion loss (25%). "
                             "Seasonal pattern — worse in summer (humidity).",
        "objectives": "Immediate containment: enhanced final inspection. Root cause: identify process parameter "
                      "link to defects. Long-term: reduce claim rate to <0.3%.",
        "primary_metric": "Claim Rate (%)", "primary_baseline": 0.8, "primary_target": 0.3,
        "secondary_metric": "Monthly Warranty Cost ($)", "secondary_baseline": 3360, "secondary_target": 1260,
        "process_start": "Parts loaded on paint line conveyor",
        "process_end": "Final visual inspection and pack",
        "excluded": "Upstream surface prep (grinding, deburring), packaging damage in transit",
        "process_owner": "Paint Line Supervisor", "sponsors": "K. Nguyen (Quality Mgr)",
        "team_members": [
            {"name": "Pat O'Brien", "role": "Project Lead", "department": "Quality Engineering"},
            {"name": "M. Harris", "role": "Lab Technician", "department": "Quality Lab"},
        ],
    },
    baselines=make_baselines(
        [0.9, 0.7, 0.8, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.8, 0.8],
        (400000, 440000), 1, "%"
    ),
    actuals=[
        {"month": 1, "baseline": 0.8, "actual": 0.75, "volume": 420000, "cost_per_unit": 1, "savings": 210, "improvement_pct": 6.3},
        {"month": 2, "baseline": 0.8, "actual": 0.70, "volume": 415000, "cost_per_unit": 1, "savings": 415, "improvement_pct": 12.5},
        {"month": 3, "baseline": 0.8, "actual": 0.65, "volume": 425000, "cost_per_unit": 1, "savings": 638, "improvement_pct": 18.8},
        {"month": 4, "baseline": 0.8, "actual": 0.60, "volume": 418000, "cost_per_unit": 1, "savings": 836, "improvement_pct": 25.0},
    ],
    actions=[
        {"title": "Deploy 100% visual inspection at end of paint line (containment)", "owner": "Pat O'Brien",
         "status": "completed", "start": date(2026,1,27), "end": date(2026,1,27), "due": date(2026,1,27), "progress": 100},
        {"title": "Pareto of defect types from last 6 months of claim data", "owner": "Pat O'Brien",
         "status": "completed", "start": date(2026,1,28), "end": date(2026,1,29), "due": date(2026,1,31), "progress": 100},
        {"title": "Install temp/humidity loggers in paint booth (4 locations)", "owner": "M. Harris",
         "status": "completed", "start": date(2026,2,3), "end": date(2026,2,7), "due": date(2026,2,7), "progress": 100},
        {"title": "Correlate environmental data with defect occurrence", "owner": "Pat O'Brien",
         "status": "in_progress", "start": date(2026,3,1), "end": None, "due": date(2026,6,30), "progress": 30},
        {"title": "DOE on pre-treatment temp, booth humidity, and cure time", "owner": "M. Harris",
         "status": "not_started", "start": None, "end": None, "due": date(2026,8,30), "progress": 0},
    ],
)


print(f"\nEnriched {len(projects)} projects with charters, baselines, actuals, and action items.")
print(f"Total action items: {ActionItem.objects.filter(source_type='hoshin').count()}")

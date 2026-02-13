"""Report type definitions for CAPA, 8D, and future report types.

Adding a new report type requires only a new entry here — zero migrations.
"""

REPORT_TYPES = {
    "capa": {
        "name": "CAPA Report",
        "description": "Corrective and Preventive Action report for addressing nonconformances.",
        "sections": [
            {
                "key": "problem_description",
                "label": "Problem Description",
                "help": "Describe the nonconformance, defect, or issue. Include when, where, and how it was detected.",
                "import_sources": ["project", "dsw", "whiteboard"],
            },
            {
                "key": "root_cause_analysis",
                "label": "Root Cause Analysis",
                "help": "Document the investigation. What is the root cause? Include 5-Why, fishbone, or other analysis.",
                "import_sources": ["hypothesis", "rca", "dsw", "whiteboard"],
            },
            {
                "key": "corrective_actions",
                "label": "Corrective Actions",
                "help": "What actions will fix the immediate problem? Include owners and target dates.",
                "import_sources": ["whiteboard"],
            },
            {
                "key": "preventive_actions",
                "label": "Preventive Actions",
                "help": "What systemic changes will prevent recurrence? Process changes, training, poka-yoke.",
                "import_sources": ["whiteboard"],
            },
            {
                "key": "verification_plan",
                "label": "Verification Plan",
                "help": "How will you verify that corrective actions were effective?",
                "import_sources": [],
            },
            {
                "key": "effectiveness_check",
                "label": "Effectiveness Check",
                "help": "Document results of verification. Was the problem resolved? Include data.",
                "import_sources": [],
            },
        ],
        "layout": "single_column",
    },
    "8d": {
        "name": "8D Report",
        "description": "Eight Disciplines problem-solving report for team-based root cause analysis.",
        "sections": [
            {
                "key": "d0_preparation",
                "label": "D0: Preparation",
                "help": "Describe the symptom. Determine if 8D is warranted. Emergency response actions if needed.",
                "import_sources": ["project"],
            },
            {
                "key": "d1_team",
                "label": "D1: Team Formation",
                "help": "List team members, roles, and qualifications. Identify champion and team leader.",
                "import_sources": ["project"],
            },
            {
                "key": "d2_problem",
                "label": "D2: Problem Description",
                "help": "Define the problem using IS/IS NOT analysis. Quantify the impact.",
                "import_sources": ["project", "dsw", "whiteboard"],
            },
            {
                "key": "d3_containment",
                "label": "D3: Containment Actions",
                "help": "Interim containment actions to protect the customer while root cause is investigated.",
                "import_sources": [],
            },
            {
                "key": "d4_root_cause",
                "label": "D4: Root Cause Analysis",
                "help": "Identify all potential causes. Verify root cause(s) with data.",
                "import_sources": ["hypothesis", "rca", "dsw", "whiteboard"],
            },
            {
                "key": "d5_corrective",
                "label": "D5: Corrective Actions",
                "help": "Choose permanent corrective actions. Verify they will resolve the problem.",
                "import_sources": ["whiteboard"],
            },
            {
                "key": "d6_implementation",
                "label": "D6: Implementation & Validation",
                "help": "Implement permanent corrective actions. Validate effectiveness with data.",
                "import_sources": [],
            },
            {
                "key": "d7_preventive",
                "label": "D7: Preventive Actions",
                "help": "Modify systems, procedures, and practices to prevent recurrence.",
                "import_sources": ["whiteboard"],
            },
            {
                "key": "d8_recognition",
                "label": "D8: Team Recognition",
                "help": "Recognize team contributions. Document lessons learned.",
                "import_sources": [],
            },
        ],
        "layout": "single_column",
    },
}

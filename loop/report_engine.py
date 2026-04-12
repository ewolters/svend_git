"""Report engine — LOOP-001 §5.2.

Assembles compliance artifacts (CAPA reports, 8D, etc.) from investigation
atoms. The engineer never writes a report. The system queries the UUID
chain and maps structured data to report sections per template.

"If an engineer is formatting a document instead of thinking about root
causes, the system has failed." — LOOP-001 §16.1.8
"""

import logging
from datetime import datetime

logger = logging.getLogger("svend.loop.report")


# =============================================================================
# ATOM COLLECTOR
# =============================================================================


def collect_atoms(investigation):
    """Collect all structured atoms from an investigation's UUID chain.

    Walks: signals → investigation → entries → tool outputs → hypotheses →
    evidence → commitments → mode transitions → verification artifacts.

    Returns a dict of atom categories, each containing the raw data.
    """
    from core.models import InvestigationToolLink

    from .models import (
        Commitment,
        ForcedFailureTest,
        ModeTransition,
        ProcessConfirmation,
        Signal,
    )

    atoms = {
        "investigation": {
            "id": str(investigation.id),
            "title": investigation.title,
            "description": investigation.description,
            "status": investigation.status,
            "version": investigation.version,
            "created_at": investigation.created_at,
            "concluded_at": investigation.concluded_at,
        },
        "signals": [],
        "graph": {},
        "tool_links": [],
        "commitments": [],
        "transitions": [],
        "documents_revised": [],
        "training_records": [],
        "pc_results": [],
        "fft_results": [],
        "reflections": [],
    }

    # Source signals
    for sig in Signal.objects.filter(resolved_by_investigation=investigation):
        atoms["signals"].append(
            {
                "id": str(sig.id),
                "title": sig.title,
                "description": sig.description,
                "source_type": sig.source_type,
                "severity": sig.severity,
                "created_at": sig.created_at,
            }
        )

    # Causal graph (Synara state)
    if investigation.synara_state:
        graph = investigation.synara_state.get("graph", investigation.synara_state)
        atoms["graph"] = {
            "hypotheses": graph.get("hypotheses", {}),
            "evidence": graph.get("evidence", []),
            "links": graph.get("links", []),
        }

    # Tool links
    for link in InvestigationToolLink.objects.filter(investigation=investigation).select_related("content_type"):
        atoms["tool_links"].append(
            {
                "tool_type": link.tool_type,
                "tool_function": link.tool_function,
                "content_type": link.content_type.model,
                "object_id": str(link.object_id),
                "linked_at": link.linked_at,
            }
        )

    # Commitments and their transitions
    for cmt in Commitment.objects.filter(source_investigation=investigation).order_by("created_at"):
        cmt_data = {
            "id": str(cmt.id),
            "title": cmt.title,
            "description": cmt.description,
            "status": cmt.status,
            "transition_type": cmt.transition_type,
            "due_date": cmt.due_date,
            "fulfilled_at": cmt.fulfilled_at,
            "owner_id": str(cmt.owner_id),
        }
        atoms["commitments"].append(cmt_data)

        # Mode transitions triggered by this commitment
        for trans in ModeTransition.objects.filter(triggered_by=cmt):
            atoms["transitions"].append(
                {
                    "id": str(trans.id),
                    "type": trans.transition_type,
                    "from_mode": trans.from_mode,
                    "to_mode": trans.to_mode,
                    "created_at": trans.created_at,
                    "target_type": trans.target_content_type.model if trans.target_content_type else None,
                    "target_id": str(trans.target_object_id) if trans.target_object_id else None,
                }
            )

    # Verification artifacts linked through commitments
    fulfilled_cmts = Commitment.objects.filter(
        source_investigation=investigation,
        status=Commitment.Status.FULFILLED,
    )

    for cmt in fulfilled_cmts:
        if not cmt.target_content_type:
            continue

        model_name = cmt.target_content_type.model

        if model_name == "processconfirmation":
            try:
                pc = ProcessConfirmation.objects.prefetch_related("observation_items").get(id=cmt.target_object_id)
                atoms["pc_results"].append(
                    {
                        "id": str(pc.id),
                        "diagnosis": pc.diagnosis,
                        "pass_rate": pc.pass_rate,
                        "created_at": pc.created_at,
                        "items_count": pc.observation_items.count(),
                    }
                )
            except ProcessConfirmation.DoesNotExist:
                pass

        elif model_name == "forcedfailuretest":
            try:
                fft = ForcedFailureTest.objects.get(id=cmt.target_object_id)
                atoms["fft_results"].append(
                    {
                        "id": str(fft.id),
                        "result": fft.result,
                        "detection_count": fft.detection_count,
                        "injection_count": fft.injection_count,
                        "detection_rate": fft.detection_rate,
                        "control_tested": fft.control_being_tested,
                        "conducted_at": fft.conducted_at,
                    }
                )
            except ForcedFailureTest.DoesNotExist:
                pass

        elif model_name == "controlleddocument":
            atoms["documents_revised"].append(
                {
                    "id": str(cmt.target_object_id),
                    "commitment_title": cmt.title,
                    "fulfilled_at": cmt.fulfilled_at,
                }
            )

        elif model_name == "trainingrequirement":
            atoms["training_records"].append(
                {
                    "id": str(cmt.target_object_id),
                    "commitment_title": cmt.title,
                    "fulfilled_at": cmt.fulfilled_at,
                }
            )

    return atoms


# =============================================================================
# REPORT TEMPLATES
# =============================================================================

# Each template defines sections with:
#   - key: section identifier
#   - title: display title
#   - required: whether this section must be populated
#   - atom_sources: which atom categories feed this section
#   - renderer: function name that renders the section content


ISO_9001_CAPA_TEMPLATE = {
    "id": "iso_9001_capa",
    "name": "ISO 9001 CAPA Report",
    "standard": "ISO 9001:2015 §10.2",
    "sections": [
        {
            "key": "problem_description",
            "title": "1. Problem Description",
            "required": True,
            "atom_sources": ["investigation", "signals"],
        },
        {
            "key": "containment",
            "title": "2. Immediate Containment",
            "required": False,
            "atom_sources": ["commitments"],
        },
        {
            "key": "root_cause",
            "title": "3. Root Cause Analysis",
            "required": True,
            "atom_sources": ["graph", "tool_links"],
        },
        {
            "key": "corrective_action",
            "title": "4. Corrective Action",
            "required": True,
            "atom_sources": ["commitments", "transitions", "documents_revised"],
        },
        {
            "key": "implementation_evidence",
            "title": "5. Implementation Evidence",
            "required": True,
            "atom_sources": ["commitments", "documents_revised", "training_records"],
        },
        {
            "key": "effectiveness_verification",
            "title": "6. Effectiveness Verification",
            "required": True,
            "atom_sources": ["pc_results", "fft_results"],
        },
        {
            "key": "recurrence_prevention",
            "title": "7. Recurrence Prevention",
            "required": True,
            "atom_sources": ["commitments"],
        },
    ],
}

IATF_8D_TEMPLATE = {
    "id": "iatf_8d",
    "name": "IATF 16949 8D Report",
    "standard": "IATF 16949:2016 §10.2.3",
    "sections": [
        {
            "key": "d0_symptom",
            "title": "D0. Symptom / Emergency Response",
            "required": True,
            "atom_sources": ["signals", "investigation"],
        },
        {
            "key": "d1_team",
            "title": "D1. Team",
            "required": True,
            "atom_sources": ["investigation"],
        },
        {
            "key": "d2_problem",
            "title": "D2. Problem Definition",
            "required": True,
            "atom_sources": ["investigation", "signals"],
        },
        {
            "key": "d3_containment",
            "title": "D3. Interim Containment",
            "required": True,
            "atom_sources": ["commitments"],
        },
        {
            "key": "d4_root_cause",
            "title": "D4. Root Cause",
            "required": True,
            "atom_sources": ["graph", "tool_links"],
        },
        {
            "key": "d5_corrective",
            "title": "D5. Permanent Corrective Action",
            "required": True,
            "atom_sources": ["commitments", "transitions", "documents_revised"],
        },
        {
            "key": "d6_validation",
            "title": "D6. Implementation / Validation",
            "required": True,
            "atom_sources": ["pc_results", "fft_results", "training_records"],
        },
        {
            "key": "d7_prevention",
            "title": "D7. Preventive Action / Horizontal Deployment",
            "required": True,
            "atom_sources": ["commitments"],
        },
        {
            "key": "d8_recognition",
            "title": "D8. Team Recognition / Lessons Learned",
            "required": True,
            "atom_sources": ["investigation", "graph"],
        },
    ],
}

TEMPLATES = {
    "iso_9001_capa": ISO_9001_CAPA_TEMPLATE,
    "iatf_8d": IATF_8D_TEMPLATE,
}


# =============================================================================
# SECTION RENDERERS
# =============================================================================


def _render_problem_description(atoms):
    """Section 1: Problem description from investigation + signal source."""
    inv = atoms["investigation"]
    lines = [f"**Investigation:** {inv['title']}"]
    if inv["description"]:
        lines.append(f"\n{inv['description']}")

    if atoms["signals"]:
        lines.append("\n**Signal Source(s):**")
        for sig in atoms["signals"]:
            lines.append(f"- [{sig['severity'].upper()}] {sig['title']} ({sig['source_type'].replace('_', ' ')})")
            if sig["description"]:
                lines.append(f"  {sig['description']}")

    lines.append(f"\n**Opened:** {_fmt_date(inv['created_at'])}")
    if inv["concluded_at"]:
        lines.append(f"**Concluded:** {_fmt_date(inv['concluded_at'])}")

    return "\n".join(lines)


def _render_containment(atoms):
    """Section 2: Early commitments as containment actions."""
    early = [c for c in atoms["commitments"] if not c["transition_type"]]
    if not early:
        return None

    lines = ["**Containment Actions:**"]
    for c in early:
        status = "Completed" if c["status"] == "fulfilled" else c["status"]
        lines.append(f"- {c['title']} — {status}")
    return "\n".join(lines)


def _render_root_cause(atoms):
    """Section 3: Root cause from causal graph hypotheses."""
    graph = atoms.get("graph", {})
    hypotheses = graph.get("hypotheses", {})

    lines = []

    if hypotheses:
        lines.append("**Hypotheses Evaluated:**")
        for h_id, h in hypotheses.items():
            prob = h.get("probability")
            label = h.get("label", h_id)
            prob_str = f" (P={prob:.0%})" if prob is not None else ""
            status = h.get("status", "")
            lines.append(f"- {label}{prob_str} {status}")
    else:
        lines.append("*No formal hypotheses recorded in causal graph.*")

    if atoms["tool_links"]:
        lines.append("\n**Analysis Tools Used:**")
        for t in atoms["tool_links"]:
            lines.append(f"- {t['tool_type']} ({t['tool_function']})")

    return "\n".join(lines)


def _render_corrective_action(atoms):
    """Section 4: Mode transitions to Standardize."""
    standardize_cmts = [
        c for c in atoms["commitments"] if c["transition_type"] in ("revise_document", "create_document", "add_control")
    ]

    lines = []
    if standardize_cmts:
        lines.append("**Corrective Actions (Investigate → Standardize):**")
        for c in standardize_cmts:
            transition = c["transition_type"].replace("_", " ").title()
            status = "Completed" if c["status"] == "fulfilled" else c["status"]
            lines.append(f"- [{transition}] {c['title']} — {status}")
            if c["fulfilled_at"]:
                lines.append(f"  Completed: {_fmt_date(c['fulfilled_at'])}")
    else:
        lines.append("*No corrective actions recorded.*")

    if atoms["documents_revised"]:
        lines.append("\n**Documents Revised:**")
        for d in atoms["documents_revised"]:
            lines.append(f"- {d['commitment_title']} (completed {_fmt_date(d['fulfilled_at'])})")

    return "\n".join(lines)


def _render_implementation_evidence(atoms):
    """Section 5: Fulfilled commitments with linked artifacts."""
    fulfilled = [c for c in atoms["commitments"] if c["status"] == "fulfilled"]

    lines = []
    if fulfilled:
        lines.append("**Implementation Evidence:**")
        for c in fulfilled:
            lines.append(f"- {c['title']} — fulfilled {_fmt_date(c['fulfilled_at'])}")
    else:
        lines.append("*No fulfilled commitments recorded.*")

    if atoms["training_records"]:
        lines.append("\n**Training Completed:**")
        for t in atoms["training_records"]:
            lines.append(f"- {t['commitment_title']} (completed {_fmt_date(t['fulfilled_at'])})")

    return "\n".join(lines)


def _render_effectiveness_verification(atoms):
    """Section 6: Verification artifacts (PCs, FFTs, SPC)."""
    lines = []

    if atoms["pc_results"]:
        lines.append("**Process Confirmation Results:**")
        for pc in atoms["pc_results"]:
            rate = f"{pc['pass_rate']:.0%}" if pc["pass_rate"] is not None else "N/A"
            lines.append(
                f"- Diagnosis: {pc['diagnosis'].replace('_', ' ')} | Pass rate: {rate} | Items: {pc['items_count']}"
            )
    else:
        lines.append("*No process confirmation data.*")

    if atoms["fft_results"]:
        lines.append("\n**Forced Failure Test Results:**")
        for fft in atoms["fft_results"]:
            rate = f"{fft['detection_rate']:.0%}" if fft["detection_rate"] is not None else "N/A"
            lines.append(f"- {fft['detection_count']}/{fft['injection_count']} detected ({rate}) — {fft['result']}")
            if fft["control_tested"]:
                lines.append(f"  Control tested: {fft['control_tested']}")
    else:
        lines.append("\n*No forced failure test data.*")

    return "\n".join(lines)


def _render_recurrence_prevention(atoms):
    """Section 7: Ongoing verification (PCs, monitoring, FFT schedule)."""
    verify_cmts = [
        c
        for c in atoms["commitments"]
        if c["transition_type"] in ("process_confirmation", "forced_failure", "monitor", "audit_zone")
    ]

    lines = []
    if verify_cmts:
        lines.append("**Recurrence Prevention (Standardize → Verify):**")
        for c in verify_cmts:
            transition = c["transition_type"].replace("_", " ").title()
            status = "Active" if c["status"] == "fulfilled" else c["status"]
            lines.append(f"- [{transition}] {c['title']} — {status}")
    else:
        lines.append("*No ongoing verification activities scheduled.*")

    lines.append(
        "\n*The Investigate → Standardize → Verify loop provides continuous recurrence prevention through process confirmations, forced failure testing, and SPC monitoring.*"
    )

    return "\n".join(lines)


# 8D-specific renderers


def _render_d0_symptom(atoms):
    return _render_problem_description(atoms)


def _render_d1_team(atoms):
    """D1: Investigation team."""
    inv = atoms["investigation"]
    # Team data would come from InvestigationMembership — for now, note the owner
    return f"**Investigation Lead:** User {inv.get('id', 'N/A')[:8]}\n\n*Full team membership available in investigation record.*"


def _render_d2_problem(atoms):
    return _render_problem_description(atoms)


def _render_d3_containment(atoms):
    return _render_containment(atoms) or "*No interim containment actions recorded.*"


def _render_d4_root_cause(atoms):
    return _render_root_cause(atoms)


def _render_d5_corrective(atoms):
    return _render_corrective_action(atoms)


def _render_d6_validation(atoms):
    return _render_effectiveness_verification(atoms)


def _render_d7_prevention(atoms):
    return _render_recurrence_prevention(atoms)


def _render_d8_recognition(atoms):
    """D8: Lessons learned."""
    graph = atoms.get("graph", {})
    hypotheses = graph.get("hypotheses", {})
    lines = ["**Lessons Learned:**"]

    if hypotheses:
        concluded = {k: v for k, v in hypotheses.items() if v.get("probability", 0) > 0.7}
        if concluded:
            for h_id, h in concluded.items():
                lines.append(f"- {h.get('label', h_id)}: confirmed with P={h['probability']:.0%}")

    lines.append("\n*Knowledge captured in process model. Causal relationships updated.*")
    return "\n".join(lines)


# Renderer dispatch
SECTION_RENDERERS = {
    "problem_description": _render_problem_description,
    "containment": _render_containment,
    "root_cause": _render_root_cause,
    "corrective_action": _render_corrective_action,
    "implementation_evidence": _render_implementation_evidence,
    "effectiveness_verification": _render_effectiveness_verification,
    "recurrence_prevention": _render_recurrence_prevention,
    # 8D
    "d0_symptom": _render_d0_symptom,
    "d1_team": _render_d1_team,
    "d2_problem": _render_d2_problem,
    "d3_containment": _render_d3_containment,
    "d4_root_cause": _render_d4_root_cause,
    "d5_corrective": _render_d5_corrective,
    "d6_validation": _render_d6_validation,
    "d7_prevention": _render_d7_prevention,
    "d8_recognition": _render_d8_recognition,
}


# =============================================================================
# REPORT ASSEMBLER
# =============================================================================


def assemble_report(investigation, template_id="iso_9001_capa"):
    """Assemble a complete report from investigation atoms.

    Returns dict with:
    - template: template metadata
    - sections: list of {key, title, content, populated, required}
    - completeness: float 0-1 (fraction of required sections populated)
    - markdown: full rendered report as markdown string
    - generated_at: timestamp
    """
    template = TEMPLATES.get(template_id)
    if not template:
        raise ValueError(f"Unknown template: {template_id}. Available: {list(TEMPLATES.keys())}")

    atoms = collect_atoms(investigation)

    sections = []
    required_count = 0
    populated_required = 0

    for section_def in template["sections"]:
        renderer = SECTION_RENDERERS.get(section_def["key"])
        content = None

        if renderer:
            try:
                content = renderer(atoms)
            except Exception as e:
                logger.error("Report renderer %s failed: %s", section_def["key"], e)
                content = f"*Error rendering section: {e}*"

        populated = content is not None and content.strip() != ""

        if section_def["required"]:
            required_count += 1
            if populated:
                populated_required += 1

        sections.append(
            {
                "key": section_def["key"],
                "title": section_def["title"],
                "content": content or "",
                "populated": populated,
                "required": section_def["required"],
                "atom_sources": section_def["atom_sources"],
            }
        )

    completeness = populated_required / required_count if required_count > 0 else 1.0

    # Render full markdown
    md_lines = [
        f"# {template['name']}",
        f"**Standard:** {template['standard']}",
        f"**Investigation:** {atoms['investigation']['title']}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Completeness:** {completeness:.0%}",
        "",
        "---",
        "",
    ]

    for section in sections:
        status = "" if section["populated"] else " *(MISSING)*"
        md_lines.append(f"## {section['title']}{status}")
        md_lines.append("")
        md_lines.append(section["content"] if section["content"] else "*No data available for this section.*")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_lines.append("*Report auto-generated by SVEND from investigation data. LOOP-001 §5.2.*")

    return {
        "template": {
            "id": template["id"],
            "name": template["name"],
            "standard": template["standard"],
        },
        "sections": sections,
        "completeness": round(completeness, 3),
        "markdown": "\n".join(md_lines),
        "generated_at": datetime.now().isoformat(),
        "investigation_id": str(investigation.id),
    }


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_date(dt):
    """Format a datetime for report display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    return str(dt)

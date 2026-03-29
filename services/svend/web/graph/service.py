"""
GraphService — GRAPH-001 §11.

The unified interface that all surfaces call. One interface, one truth.
Every method takes tenant_id as explicit parameter (O2 decision).
"""

import logging
from datetime import datetime
from uuid import UUID

from django.db import models as db_models
from django.db import transaction
from django.utils import timezone

from .models import EdgeEvidence, ProcessEdge, ProcessGraph, ProcessNode

logger = logging.getLogger("svend.graph")


class GapReport:
    """Result of gap analysis across a graph."""

    def __init__(self):
        self.uncalibrated_edges: list[ProcessEdge] = []
        self.stale_edges: list[ProcessEdge] = []
        self.contradicted_edges: list[ProcessEdge] = []
        self.uncalibrated_interactions: list[ProcessEdge] = []
        self.measurement_gaps: list[ProcessNode] = []
        self.low_confidence_edges: list[ProcessEdge] = []

    @property
    def total_gaps(self) -> int:
        return (
            len(self.uncalibrated_edges)
            + len(self.stale_edges)
            + len(self.contradicted_edges)
            + len(self.uncalibrated_interactions)
            + len(self.measurement_gaps)
            + len(self.low_confidence_edges)
        )

    def to_dict(self) -> dict:
        return {
            "total_gaps": self.total_gaps,
            "uncalibrated_edges": [str(e.id) for e in self.uncalibrated_edges],
            "stale_edges": [str(e.id) for e in self.stale_edges],
            "contradicted_edges": [str(e.id) for e in self.contradicted_edges],
            "uncalibrated_interactions": [str(e.id) for e in self.uncalibrated_interactions],
            "measurement_gaps": [str(n.id) for n in self.measurement_gaps],
            "low_confidence_edges": [str(e.id) for e in self.low_confidence_edges],
        }


class WritebackResult:
    """Result of writing investigation findings back to the graph."""

    def __init__(self):
        self.created_nodes: list[ProcessNode] = []
        self.created_edges: list[ProcessEdge] = []
        self.added_evidence: list[EdgeEvidence] = []
        self.contradictions_raised: list[UUID] = []


class GraphService:
    """Unified Knowledge Graph + Process Model service.

    Every tool in the platform reads from or writes to the graph
    through this service. There is no other path to graph state.
    """

    # ── Graph Lifecycle ──

    @staticmethod
    def create_graph(
        tenant_id: UUID,
        name: str,
        description: str = "",
        process_area: str = "",
        parent_graph_id: UUID | None = None,
        created_by=None,
    ) -> ProcessGraph:
        return ProcessGraph.objects.create(
            tenant_id=tenant_id,
            name=name,
            description=description,
            process_area=process_area,
            parent_graph_id=parent_graph_id,
            created_by=created_by,
        )

    @staticmethod
    def get_graph(tenant_id: UUID, graph_id: UUID) -> ProcessGraph:
        return ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)

    @staticmethod
    def get_org_graph(tenant_id: UUID) -> ProcessGraph | None:
        """Get the org's primary process graph (no parent)."""
        return (
            ProcessGraph.objects.filter(tenant_id=tenant_id, parent_graph__isnull=True).order_by("created_at").first()
        )

    @staticmethod
    def get_or_create_org_graph(tenant_id: UUID, created_by=None) -> ProcessGraph:
        """Get or create the org's primary process graph."""
        graph = GraphService.get_org_graph(tenant_id)
        if graph is None:
            graph = GraphService.create_graph(
                tenant_id=tenant_id,
                name="Process Model",
                description="Organization-wide process knowledge graph",
                created_by=created_by,
            )
        return graph

    # ── Node Operations ──

    @staticmethod
    def add_node(
        tenant_id: UUID,
        graph_id: UUID,
        name: str,
        node_type: str = "process_parameter",
        description: str = "",
        created_by=None,
        **kwargs,
    ) -> ProcessNode:
        graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
        return ProcessNode.objects.create(
            graph=graph,
            name=name,
            node_type=node_type,
            description=description,
            created_by=created_by,
            **kwargs,
        )

    @staticmethod
    def update_node(tenant_id: UUID, node_id: UUID, **updates) -> ProcessNode:
        node = ProcessNode.objects.select_related("graph").get(id=node_id, graph__tenant_id=tenant_id)
        for field, value in updates.items():
            setattr(node, field, value)
        node.save(update_fields=list(updates.keys()) + ["updated_at"])
        return node

    @staticmethod
    def get_node(tenant_id: UUID, node_id: UUID) -> ProcessNode:
        return ProcessNode.objects.select_related("graph").get(id=node_id, graph__tenant_id=tenant_id)

    @staticmethod
    def get_nodes(tenant_id: UUID, graph_id: UUID, **filters) -> list[ProcessNode]:
        qs = ProcessNode.objects.filter(graph_id=graph_id, graph__tenant_id=tenant_id)
        if "node_type" in filters:
            qs = qs.filter(node_type=filters["node_type"])
        if "name__icontains" in filters:
            qs = qs.filter(name__icontains=filters["name__icontains"])
        return list(qs)

    @staticmethod
    def remove_node(tenant_id: UUID, node_id: UUID) -> None:
        ProcessNode.objects.filter(id=node_id, graph__tenant_id=tenant_id).delete()

    # ── Edge Operations ──

    @staticmethod
    def add_edge(
        tenant_id: UUID,
        graph_id: UUID,
        source_id: UUID,
        target_id: UUID,
        relation_type: str = "causal",
        provenance: str = "manual",
        created_by=None,
        **kwargs,
    ) -> ProcessEdge:
        graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
        # Verify both nodes belong to this graph (or are shared)
        source = ProcessNode.objects.get(id=source_id)
        target = ProcessNode.objects.get(id=target_id)
        if source.graph_id != graph_id and not source.shared:
            raise ValueError(f"Source node {source_id} not in graph {graph_id}")
        if target.graph_id != graph_id and not target.shared:
            raise ValueError(f"Target node {target_id} not in graph {graph_id}")

        return ProcessEdge.objects.create(
            graph=graph,
            source=source,
            target=target,
            relation_type=relation_type,
            provenance=provenance,
            created_by=created_by,
            **kwargs,
        )

    @staticmethod
    def get_edge(tenant_id: UUID, edge_id: UUID) -> ProcessEdge:
        return ProcessEdge.objects.select_related("source", "target", "graph").get(
            id=edge_id, graph__tenant_id=tenant_id
        )

    @staticmethod
    def get_edges(tenant_id: UUID, graph_id: UUID, **filters) -> list[ProcessEdge]:
        qs = ProcessEdge.objects.filter(graph_id=graph_id, graph__tenant_id=tenant_id).select_related(
            "source", "target"
        )
        if "relation_type" in filters:
            qs = qs.filter(relation_type=filters["relation_type"])
        if "is_stale" in filters:
            qs = qs.filter(is_stale=filters["is_stale"])
        if "is_contradicted" in filters:
            qs = qs.filter(is_contradicted=filters["is_contradicted"])
        return list(qs)

    @staticmethod
    def remove_edge(tenant_id: UUID, edge_id: UUID) -> None:
        ProcessEdge.objects.filter(id=edge_id, graph__tenant_id=tenant_id).delete()

    # ── Evidence (the core operation) ──

    @staticmethod
    @transaction.atomic
    def add_evidence(
        tenant_id: UUID,
        edge_id: UUID,
        source_type: str,
        observed_at: datetime,
        source_description: str = "",
        effect_size: float | None = None,
        confidence_interval: dict | None = None,
        sample_size: int | None = None,
        p_value: float | None = None,
        strength: float = 1.0,
        source_id: UUID | None = None,
        created_by=None,
    ) -> EdgeEvidence:
        """Add evidence to an edge.

        This is the primary write operation. Everything in the platform
        that learns something about a process relationship calls this.
        Updates edge posterior_strength and evidence_count.
        """
        edge = ProcessEdge.objects.select_for_update().get(id=edge_id, graph__tenant_id=tenant_id)

        evidence = EdgeEvidence.objects.create(
            edge=edge,
            source_type=source_type,
            observed_at=observed_at,
            source_description=source_description,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            sample_size=sample_size,
            p_value=p_value,
            strength=strength,
            source_id=source_id,
            created_by=created_by,
        )

        # Update edge denormalized count
        edge.evidence_count = edge.evidence_stack.filter(retracted=False).count()

        # Update edge effect size if evidence provides one
        if effect_size is not None:
            edge.effect_size = effect_size
            edge.calibration_date = observed_at
            if edge.provenance == "fmea_assertion":
                edge.provenance = source_type
        if confidence_interval:
            edge.effect_ci_lower = confidence_interval.get("lower")
            edge.effect_ci_upper = confidence_interval.get("upper")

        # Recalculate posterior from evidence stack
        edge.posterior_strength = GraphService._recompute_posterior(edge)

        edge.save()

        logger.info(
            "Evidence added to edge %s (%s): effect_size=%s, strength=%s",
            edge_id,
            source_type,
            effect_size,
            strength,
        )

        return evidence

    @staticmethod
    def _recompute_posterior(edge: ProcessEdge, half_life_days: int = 180) -> float:
        """Recency-weighted posterior from evidence stack.

        Implements GRAPH-001 §4.4: more recent evidence weighs more,
        but old evidence is never discarded. Exponential decay with
        configurable half-life.
        """
        import math

        evidence_qs = edge.evidence_stack.filter(retracted=False).order_by("observed_at")
        records = list(evidence_qs)

        if not records:
            return 0.5  # uninformative prior

        now = timezone.now()
        decay_rate = math.log(2) / half_life_days if half_life_days > 0 else 0

        weighted_sum = 0.0
        weight_total = 0.0

        for ev in records:
            age_days = (now - ev.observed_at).total_seconds() / 86400
            recency_weight = math.exp(-decay_rate * age_days) if decay_rate > 0 else 1.0
            w = ev.strength * recency_weight

            # Evidence contributes toward confidence (0.5 = unknown, 1.0 = certain)
            # Higher effect_size + lower p_value + higher strength = more confidence
            if ev.effect_size is not None:
                # Normalize: large effect sizes push posterior higher
                contribution = min(1.0, 0.5 + abs(ev.effect_size) * 0.1)
            elif ev.p_value is not None and ev.p_value < 0.05:
                contribution = 0.8
            else:
                contribution = 0.6  # weak but confirmatory

            weighted_sum += w * contribution
            weight_total += w

        if weight_total == 0:
            return 0.5

        return max(0.01, min(0.99, weighted_sum / weight_total))

    @staticmethod
    def retract_evidence(
        tenant_id: UUID,
        evidence_id: UUID,
        reason: str,
    ) -> EdgeEvidence:
        """Retract erroneous evidence (O1 decision)."""
        ev = EdgeEvidence.objects.select_related("edge__graph").get(id=evidence_id, edge__graph__tenant_id=tenant_id)
        ev.retracted = True
        ev.retracted_reason = reason
        ev.save(update_fields=["retracted", "retracted_reason"])

        # Recompute edge posterior without retracted evidence
        edge = ev.edge
        edge.evidence_count = edge.evidence_stack.filter(retracted=False).count()
        edge.posterior_strength = GraphService._recompute_posterior(edge)
        edge.save(update_fields=["evidence_count", "posterior_strength", "updated_at"])

        return ev

    # ── SPC Integration ──

    @staticmethod
    def update_node_distribution(
        tenant_id: UUID,
        node_id: UUID,
        distribution: dict,
    ) -> ProcessNode:
        node = ProcessNode.objects.select_related("graph").get(id=node_id, graph__tenant_id=tenant_id)
        node.distribution = distribution
        node.save(update_fields=["distribution", "updated_at"])
        return node

    @staticmethod
    def flag_stale_edges(
        tenant_id: UUID,
        node_id: UUID,
        reason: str = "spc_shift_detected",
    ) -> list[ProcessEdge]:
        """Flag all edges touching a node as stale."""
        node = ProcessNode.objects.select_related("graph").get(id=node_id, graph__tenant_id=tenant_id)
        edges = ProcessEdge.objects.filter(
            graph=node.graph,
        ).filter(db_models.Q(source=node) | db_models.Q(target=node))
        stale_edges = []
        for edge in edges:
            if not edge.is_stale:
                edge.is_stale = True
                edge.staleness_reason = reason
                edge.save(update_fields=["is_stale", "staleness_reason", "updated_at"])
                stale_edges.append(edge)
        return stale_edges

    # ── Gap Analysis ──

    @staticmethod
    def gap_report(tenant_id: UUID, graph_id: UUID, confidence_threshold: float = 0.3) -> GapReport:
        """Full gap taxonomy across the graph — GRAPH-001 §6."""
        report = GapReport()

        edges = ProcessEdge.objects.filter(graph_id=graph_id, graph__tenant_id=tenant_id).select_related(
            "source", "target"
        )

        nodes = ProcessNode.objects.filter(graph_id=graph_id, graph__tenant_id=tenant_id)

        for edge in edges:
            # Uncalibrated: FMEA assertion with no evidence
            if edge.evidence_count == 0 and edge.provenance == "fmea_assertion":
                report.uncalibrated_edges.append(edge)

            # Stale
            if edge.is_stale:
                report.stale_edges.append(edge)

            # Contradicted
            if edge.is_contradicted:
                report.contradicted_edges.append(edge)

            # Uncalibrated interactions
            for term in edge.interaction_terms or []:
                if not term.get("calibrated", False):
                    report.uncalibrated_interactions.append(edge)
                    break

            # Low confidence
            if edge.posterior_strength < confidence_threshold and edge.evidence_count > 0:
                report.low_confidence_edges.append(edge)

        # Measurement gaps: nodes with no linked measurement system
        for node in nodes:
            if (
                node.node_type
                in (
                    ProcessNode.NodeType.PROCESS_PARAMETER,
                    ProcessNode.NodeType.QUALITY_CHARACTERISTIC,
                )
                and not node.linked_equipment
            ):
                report.measurement_gaps.append(node)

        return report

    # ── FMIS Seeding — GRAPH-001 §7 ──

    @staticmethod
    def seed_from_fmis(
        tenant_id: UUID,
        graph_id: UUID,
        fmis_rows: list,
    ) -> list[dict]:
        """Generate proposed nodes + edges from FMIS rows.

        Returns proposals for user confirmation. Does not auto-commit.
        Each proposal has type, data, and a reference to the source FMIS row.
        """
        proposals = []

        graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
        existing_nodes = {n.name.lower(): n for n in graph.nodes.all()}

        for row in fmis_rows:
            row_proposals = []

            # Propose cause node
            cause_name = getattr(row, "cause_text", "") or ""
            if cause_name and cause_name.lower() not in existing_nodes:
                row_proposals.append(
                    {
                        "type": "new_node",
                        "data": {
                            "name": cause_name,
                            "node_type": "process_parameter",
                            "provenance": "fmea_seed",
                        },
                        "fmis_row_id": str(row.id),
                        "role": "cause",
                    }
                )

            # Propose failure mode node
            fm_name = getattr(row, "failure_mode_text", "") or ""
            if fm_name and fm_name.lower() not in existing_nodes:
                row_proposals.append(
                    {
                        "type": "new_node",
                        "data": {
                            "name": fm_name,
                            "node_type": "failure_mode",
                            "provenance": "fmea_seed",
                        },
                        "fmis_row_id": str(row.id),
                        "role": "failure_mode",
                    }
                )

            # Propose effect node
            effect_name = getattr(row, "effect_text", "") or ""
            if effect_name and effect_name.lower() not in existing_nodes:
                row_proposals.append(
                    {
                        "type": "new_node",
                        "data": {
                            "name": effect_name,
                            "node_type": "quality_characteristic",
                            "provenance": "fmea_seed",
                        },
                        "fmis_row_id": str(row.id),
                        "role": "effect",
                    }
                )

            # Propose edges: cause → failure_mode, failure_mode → effect
            if cause_name and fm_name:
                row_proposals.append(
                    {
                        "type": "new_edge",
                        "data": {
                            "source_name": cause_name,
                            "target_name": fm_name,
                            "relation_type": "causal",
                            "provenance": "fmea_assertion",
                        },
                        "fmis_row_id": str(row.id),
                    }
                )
            if fm_name and effect_name:
                row_proposals.append(
                    {
                        "type": "new_edge",
                        "data": {
                            "source_name": fm_name,
                            "target_name": effect_name,
                            "relation_type": "causal",
                            "provenance": "fmea_assertion",
                        },
                        "fmis_row_id": str(row.id),
                    }
                )

            proposals.extend(row_proposals)

        return proposals

    @staticmethod
    @transaction.atomic
    def confirm_seed(
        tenant_id: UUID,
        graph_id: UUID,
        proposals: list[dict],
        created_by=None,
    ) -> dict:
        """Confirm proposed nodes + edges from FMIS seeding.

        Returns created nodes and edges.
        """
        graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
        created_nodes = []
        created_edges = []

        # First pass: create nodes
        node_map = {n.name.lower(): n for n in graph.nodes.all()}
        for p in proposals:
            if p["type"] == "new_node":
                name = p["data"]["name"]
                if name.lower() not in node_map:
                    node = ProcessNode.objects.create(
                        graph=graph,
                        name=name,
                        node_type=p["data"].get("node_type", "process_parameter"),
                        provenance=p["data"].get("provenance", "fmea_seed"),
                        created_by=created_by,
                    )
                    node_map[name.lower()] = node
                    created_nodes.append(node)

        # Second pass: create edges
        for p in proposals:
            if p["type"] == "new_edge":
                source_name = p["data"]["source_name"]
                target_name = p["data"]["target_name"]
                source = node_map.get(source_name.lower())
                target = node_map.get(target_name.lower())
                if source and target:
                    # Check if edge already exists
                    exists = ProcessEdge.objects.filter(graph=graph, source=source, target=target).exists()
                    if not exists:
                        edge = ProcessEdge.objects.create(
                            graph=graph,
                            source=source,
                            target=target,
                            relation_type=p["data"].get("relation_type", "causal"),
                            provenance=p["data"].get("provenance", "fmea_assertion"),
                            created_by=created_by,
                        )
                        created_edges.append(edge)

        return {
            "created_nodes": created_nodes,
            "created_edges": created_edges,
        }

    # ── Queries ──

    @staticmethod
    def get_upstream(tenant_id: UUID, node_id: UUID, depth: int | None = None) -> list[ProcessNode]:
        """Get all nodes upstream of the given node."""
        node = ProcessNode.objects.get(id=node_id, graph__tenant_id=tenant_id)
        visited = set()
        result = []
        GraphService._traverse_upstream(node, visited, result, depth, 0)
        return result

    @staticmethod
    def _traverse_upstream(node, visited, result, max_depth, current_depth):
        if max_depth is not None and current_depth >= max_depth:
            return
        for edge in ProcessEdge.objects.filter(target=node).select_related("source"):
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                result.append(edge.source)
                GraphService._traverse_upstream(edge.source, visited, result, max_depth, current_depth + 1)

    @staticmethod
    def get_downstream(tenant_id: UUID, node_id: UUID, depth: int | None = None) -> list[ProcessNode]:
        """Get all nodes downstream of the given node."""
        node = ProcessNode.objects.get(id=node_id, graph__tenant_id=tenant_id)
        visited = set()
        result = []
        GraphService._traverse_downstream(node, visited, result, depth, 0)
        return result

    @staticmethod
    def _traverse_downstream(node, visited, result, max_depth, current_depth):
        if max_depth is not None and current_depth >= max_depth:
            return
        for edge in ProcessEdge.objects.filter(source=node).select_related("target"):
            if edge.target_id not in visited:
                visited.add(edge.target_id)
                result.append(edge.target)
                GraphService._traverse_downstream(edge.target, visited, result, max_depth, current_depth + 1)

    @staticmethod
    def explain_edge(tenant_id: UUID, edge_id: UUID) -> dict:
        """Full evidence history + posterior reasoning for an edge."""
        edge = ProcessEdge.objects.select_related("source", "target", "graph").get(
            id=edge_id, graph__tenant_id=tenant_id
        )
        evidence = list(
            edge.evidence_stack.order_by("-observed_at").values(
                "id",
                "source_type",
                "effect_size",
                "confidence_interval",
                "sample_size",
                "p_value",
                "strength",
                "observed_at",
                "source_description",
                "retracted",
                "retracted_reason",
            )
        )
        return {
            "edge_id": str(edge.id),
            "source": {"id": str(edge.source.id), "name": edge.source.name},
            "target": {"id": str(edge.target.id), "name": edge.target.name},
            "relation_type": edge.relation_type,
            "effect_size": edge.effect_size,
            "effect_ci": {"lower": edge.effect_ci_lower, "upper": edge.effect_ci_upper},
            "posterior_strength": edge.posterior_strength,
            "direction": edge.direction,
            "provenance": edge.provenance,
            "calibration_date": edge.calibration_date.isoformat() if edge.calibration_date else None,
            "is_calibrated": edge.is_calibrated,
            "is_stale": edge.is_stale,
            "staleness_reason": edge.staleness_reason,
            "is_contradicted": edge.is_contradicted,
            "evidence_count": edge.evidence_count,
            "evidence": evidence,
            "interaction_terms": edge.interaction_terms,
        }

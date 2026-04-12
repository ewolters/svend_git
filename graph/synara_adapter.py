"""
Synara adapter — GRAPH-001 §12.4, §13.3.

Bridges Django models (ProcessGraph/ProcessNode/ProcessEdge) to Synara's
in-memory CausalGraph for Bayesian computation. Load, compute, persist.
No long-lived in-memory graph state.

Round-trip: Django models → CausalGraph → Synara operations → Django models.
"""

import logging
from uuid import UUID

from agents_api.synara.belief import BeliefEngine
from agents_api.synara.kernel import (
    CausalGraph,
    CausalLink,
    Evidence,
    HypothesisRegion,
)

from .models import EdgeEvidence, ProcessEdge, ProcessGraph

logger = logging.getLogger("svend.graph.synara")


class SynaraAdapter:
    """Bidirectional bridge between ProcessGraph (Django) and CausalGraph (Synara)."""

    @staticmethod
    def load_causal_graph(
        graph_id: UUID,
        tenant_id: UUID,
        node_ids: list[UUID] | None = None,
    ) -> tuple[CausalGraph, dict[str, UUID], dict[str, UUID]]:
        """Load a ProcessGraph (or subgraph) into a Synara CausalGraph.

        Args:
            graph_id: ProcessGraph to load
            tenant_id: Tenant for isolation
            node_ids: Optional subset of nodes. If None, loads entire graph.

        Returns:
            (causal_graph, node_id_map, edge_id_map)
            - node_id_map: {synara_h_id: ProcessNode UUID}
            - edge_id_map: {f"{from_id}->{to_id}": ProcessEdge UUID}
        """
        graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)

        nodes_qs = graph.nodes.all()
        if node_ids:
            nodes_qs = nodes_qs.filter(id__in=node_ids)

        nodes = list(nodes_qs)
        node_uuid_set = {n.id for n in nodes}

        edges_qs = graph.edges.select_related("source", "target").all()
        if node_ids:
            edges_qs = edges_qs.filter(source_id__in=node_uuid_set, target_id__in=node_uuid_set)

        edges = list(edges_qs)

        cg = CausalGraph()
        node_id_map = {}  # synara_id → ProcessNode UUID
        edge_id_map = {}  # "from->to" → ProcessEdge UUID

        # Map nodes → HypothesisRegions
        for node in nodes:
            h_id = str(node.id)
            node_id_map[h_id] = node.id

            h = HypothesisRegion(
                id=h_id,
                description=node.name,
                domain_conditions={
                    "node_type": node.node_type,
                    "unit": node.unit,
                },
                behavior_class=node.node_type,
                prior=0.5,
                posterior=0.5,
                source="graph",
            )
            cg.add_hypothesis(h)

        # Map edges → CausalLinks
        for edge in edges:
            from_id = str(edge.source_id)
            to_id = str(edge.target_id)

            if from_id not in cg.hypotheses or to_id not in cg.hypotheses:
                continue

            link = CausalLink(
                from_id=from_id,
                to_id=to_id,
                mechanism=f"{edge.source.name} → {edge.target.name}",
                strength=edge.posterior_strength,
                relation="contributes",
                source=edge.provenance,
            )
            cg.add_link(link)
            edge_id_map[f"{from_id}->{to_id}"] = edge.id

        return cg, node_id_map, edge_id_map

    @staticmethod
    def persist_posteriors(
        causal_graph: CausalGraph,
        edge_id_map: dict[str, UUID],
        tenant_id: UUID,
    ) -> int:
        """Write Synara posteriors back to ProcessEdge records.

        Returns count of edges updated.
        """
        updated = 0
        for link in causal_graph.links:
            key = f"{link.from_id}->{link.to_id}"
            edge_uuid = edge_id_map.get(key)
            if not edge_uuid:
                continue

            # Synara's link.strength is the propagated belief
            try:
                edge = ProcessEdge.objects.get(id=edge_uuid, graph__tenant_id=tenant_id)
            except ProcessEdge.DoesNotExist:
                continue

            if abs(edge.posterior_strength - link.strength) > 0.001:
                edge.posterior_strength = link.strength
                edge.save(update_fields=["posterior_strength", "updated_at"])
                updated += 1

        return updated

    @staticmethod
    def add_evidence_and_propagate(
        graph_id: UUID,
        tenant_id: UUID,
        edge_id: UUID,
        evidence: Evidence,
    ) -> dict[str, float]:
        """Add evidence to an edge and propagate belief through the graph.

        This is the full Synara cycle:
        1. Load graph into CausalGraph
        2. Find the target hypothesis (edge target node)
        3. Add evidence via BeliefEngine
        4. Propagate changes
        5. Check for expansion signals
        6. Persist updated posteriors

        Returns dict of {node_id: new_posterior} for changed nodes.
        """
        cg, node_id_map, edge_id_map = SynaraAdapter.load_causal_graph(graph_id, tenant_id)

        engine = BeliefEngine()

        # Compute likelihoods
        likelihoods = engine.compute_all_likelihoods(cg, evidence)

        # Check expansion signal (whole-graph)
        expansion = engine.check_expansion(evidence, likelihoods)
        if expansion:
            logger.warning(
                "Expansion signal on graph %s: %s",
                graph_id,
                expansion.message,
            )

        # Update posteriors
        engine.update_posteriors(cg, evidence, likelihoods)

        # Propagate through graph
        all_changes = {}
        for h_id in cg.hypotheses:
            changes = engine.propagate_belief(cg, h_id)
            all_changes.update(changes)

        # Persist back to Django models
        SynaraAdapter.persist_posteriors(cg, edge_id_map, tenant_id)

        return all_changes

    @staticmethod
    def check_edge_contradiction(
        tenant_id: UUID,
        edge_id: UUID,
        new_evidence: EdgeEvidence,
        threshold: float = 0.05,
    ) -> dict | None:
        """Check if new evidence contradicts an edge's current posterior.

        Edge-scoped expansion signal — GRAPH-001 §10, conference D8.
        Same math as Synara's check_expansion() but scoped to one edge.

        Returns contradiction signal dict if detected, None otherwise.
        """
        edge = ProcessEdge.objects.select_related("source", "target").get(id=edge_id, graph__tenant_id=tenant_id)

        if edge.evidence_count < 2:
            return None

        # Build a minimal Synara evidence object
        synara_evidence = Evidence(
            id=str(new_evidence.id),
            event=f"evidence_{new_evidence.source_type}",
            context={
                "effect_size": new_evidence.effect_size,
                "p_value": new_evidence.p_value,
                "source_type": new_evidence.source_type,
            },
            strength=new_evidence.strength,
            source=new_evidence.source_type,
        )

        # Build minimal hypothesis for the edge's target
        h = HypothesisRegion(
            id=str(edge.target_id),
            description=edge.target.name,
            posterior=edge.posterior_strength,
        )

        engine = BeliefEngine()
        likelihood = engine.compute_likelihood(synara_evidence, h)

        if likelihood < threshold:
            signal = {
                "type": "edge_contradiction",
                "edge_id": str(edge.id),
                "source_node": edge.source.name,
                "target_node": edge.target.name,
                "current_posterior": edge.posterior_strength,
                "evidence_likelihood": likelihood,
                "threshold": threshold,
                "evidence_id": str(new_evidence.id),
                "message": (
                    f"Evidence on {edge.source.name} → {edge.target.name} "
                    f"has low likelihood ({likelihood:.3f}) under current posterior "
                    f"({edge.posterior_strength:.3f}). Possible contradiction."
                ),
            }
            logger.warning("Contradiction detected: %s", signal["message"])
            return signal

        return None

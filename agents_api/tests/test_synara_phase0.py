"""
Tests for Object 271 Phase 0 Synara safety fixes.

Covers:
- Cycle detection in CausalGraph.add_link()
- Visited set in propagate_belief()
- from_dict() non-mutation guarantee
"""

from django.test import SimpleTestCase

from agents_api.synara.belief import BeliefEngine
from agents_api.synara.kernel import (
    CausalGraph,
    CausalLink,
    Evidence,
    HypothesisRegion,
)


class CycleDetectionTest(SimpleTestCase):
    """CausalGraph.add_link() must reject links that would create cycles."""

    def _make_graph(self):
        g = CausalGraph()
        g.add_hypothesis(HypothesisRegion(id="a", description="A"))
        g.add_hypothesis(HypothesisRegion(id="b", description="B"))
        g.add_hypothesis(HypothesisRegion(id="c", description="C"))
        return g

    def test_simple_chain_allowed(self):
        g = self._make_graph()
        g.add_link(CausalLink(from_id="a", to_id="b"))
        g.add_link(CausalLink(from_id="b", to_id="c"))
        self.assertEqual(len(g.links), 2)

    def test_direct_cycle_rejected(self):
        g = self._make_graph()
        g.add_link(CausalLink(from_id="a", to_id="b"))
        with self.assertRaises(ValueError, msg="a→b→a should be rejected"):
            g.add_link(CausalLink(from_id="b", to_id="a"))

    def test_indirect_cycle_rejected(self):
        g = self._make_graph()
        g.add_link(CausalLink(from_id="a", to_id="b"))
        g.add_link(CausalLink(from_id="b", to_id="c"))
        with self.assertRaises(ValueError, msg="c→a should be rejected (a→b→c→a cycle)"):
            g.add_link(CausalLink(from_id="c", to_id="a"))

    def test_self_loop_rejected(self):
        g = self._make_graph()
        with self.assertRaises(ValueError, msg="a→a should be rejected"):
            g.add_link(CausalLink(from_id="a", to_id="a"))

    def test_parallel_edges_allowed(self):
        """Two edges between same pair in same direction are allowed."""
        g = self._make_graph()
        g.add_link(CausalLink(from_id="a", to_id="b", mechanism="direct"))
        g.add_link(CausalLink(from_id="a", to_id="b", mechanism="indirect"))
        self.assertEqual(len(g.links), 2)

    def test_diamond_allowed(self):
        """a→b, a→c, b→d, c→d is a valid DAG (diamond shape)."""
        g = CausalGraph()
        for nid in ("a", "b", "c", "d"):
            g.add_hypothesis(HypothesisRegion(id=nid, description=nid))
        g.add_link(CausalLink(from_id="a", to_id="b"))
        g.add_link(CausalLink(from_id="a", to_id="c"))
        g.add_link(CausalLink(from_id="b", to_id="d"))
        g.add_link(CausalLink(from_id="c", to_id="d"))
        self.assertEqual(len(g.links), 4)


class PropagateBeliefVisitedSetTest(SimpleTestCase):
    """propagate_belief() must terminate even if graph state is inconsistent."""

    def test_propagation_terminates_on_chain(self):
        """a→b→c should propagate without infinite recursion."""
        g = CausalGraph()
        g.add_hypothesis(HypothesisRegion(id="a", description="A", prior=0.8, posterior=0.8))
        g.add_hypothesis(HypothesisRegion(id="b", description="B", prior=0.5, posterior=0.5))
        g.add_hypothesis(HypothesisRegion(id="c", description="C", prior=0.5, posterior=0.5))
        g.add_link(CausalLink(from_id="a", to_id="b", strength=0.7))
        g.add_link(CausalLink(from_id="b", to_id="c", strength=0.7))

        engine = BeliefEngine()
        changes = engine.propagate_belief(g, "a")
        self.assertIn("b", changes)

    def test_visited_set_prevents_revisit(self):
        """Manually corrupt graph with cycle in downstream refs.
        propagate_belief must still terminate."""
        g = CausalGraph()
        g.add_hypothesis(HypothesisRegion(id="x", description="X", prior=0.8, posterior=0.8))
        g.add_hypothesis(HypothesisRegion(id="y", description="Y", prior=0.5, posterior=0.5))
        # Add link normally
        g.add_link(CausalLink(from_id="x", to_id="y", strength=0.7))
        # Corrupt: manually add y→x to downstream refs (bypassing add_link validation)
        g.hypotheses["y"].downstream.append("x")
        g.hypotheses["x"].upstream.append("y")
        g.links.append(CausalLink(from_id="y", to_id="x", strength=0.5))

        engine = BeliefEngine()
        # This would infinite-recurse without the visited set
        changes = engine.propagate_belief(g, "x")
        # Should terminate and return some changes
        self.assertIsInstance(changes, dict)


class FromDictNonMutationTest(SimpleTestCase):
    """from_dict() must not mutate the input dictionary."""

    def test_hypothesis_from_dict_preserves_input(self):
        data = {
            "id": "h1",
            "description": "test",
            "created_at": "2026-03-28T12:00:00",
        }
        original_created_at = data["created_at"]
        HypothesisRegion.from_dict(data)
        self.assertEqual(
            data["created_at"],
            original_created_at,
            "from_dict must not mutate input dict",
        )

    def test_evidence_from_dict_preserves_input(self):
        data = {
            "id": "e1",
            "event": "test_event",
            "timestamp": "2026-03-28T12:00:00",
        }
        original_timestamp = data["timestamp"]
        Evidence.from_dict(data)
        self.assertEqual(
            data["timestamp"],
            original_timestamp,
            "from_dict must not mutate input dict",
        )

    def test_hypothesis_from_dict_roundtrip(self):
        h = HypothesisRegion(id="h2", description="roundtrip test", prior=0.6, posterior=0.6)
        d = h.to_dict()
        h2 = HypothesisRegion.from_dict(d)
        self.assertEqual(h2.id, "h2")
        self.assertEqual(h2.prior, 0.6)

    def test_evidence_from_dict_roundtrip(self):
        e = Evidence(id="e2", event="roundtrip_event", strength=0.9)
        d = e.to_dict()
        e2 = Evidence.from_dict(d)
        self.assertEqual(e2.id, "e2")
        self.assertEqual(e2.strength, 0.9)

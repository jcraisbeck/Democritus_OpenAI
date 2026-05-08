"""Tests for the pure-computation portion of the indiscernibility
quotient: structural_equivalence, _build_output_records, idempotence,
edge-set monotonicity, schema conformance.

These tests build small synthetic graphs by hand. None of them touch
an LLM. Run with:

    pytest Democritus_OpenAI/tests/test_structural_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Make the Democritus_OpenAI/ root importable when pytest is invoked
# from the repository root. The package layout uses bare `scripts.`
# and `configs.` imports, so we put the parent of this tests/
# directory onto sys.path.
_HERE = Path(__file__).resolve()
_DEMOCRITUS_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_DEMOCRITUS_ROOT))

from scripts.indiscernibility_quotient import (   # noqa: E402
    Edge,
    structural_equivalence,
    _build_output_records,
    _filter_uncorroborated_expansion_edges,
    _triples_to_edges,
    quotient_triples,
)


# ---------------------------------------------------------------------
# Helpers — build synthetic edges for tests
# ---------------------------------------------------------------------

def _ext_edge(subj: str, rel: str, obj: str, domain: str = "d0") -> Edge:
    """Build a synthetic extraction edge with empty provenance."""
    return Edge(subj=subj, rel=rel, obj=obj, domain=domain,
                edge_source="extraction", provenance=[])


def _exp_edge(subj: str, rel: str, obj: str, domain: str = "d0") -> Edge:
    return Edge(subj=subj, rel=rel, obj=obj, domain=domain,
                edge_source="expansion", provenance=[])


# ---------------------------------------------------------------------
# Test 1: two propositions with the SAME out- and in-neighbour sets
# (across the same relation labels) collapse into one equivalence
# class. Two propositions whose neighbour sets DIFFER stay separate.
# ---------------------------------------------------------------------

def test_indiscernibles_collapse_distinguishables_remain():
    propositions = ["A", "B", "C", "D"]
    edges = [
        # A and B both point to C via "causes"; both are pointed at by D
        # via "increases". A ~ B should hold.
        _ext_edge("A", "causes", "C"),
        _ext_edge("B", "causes", "C"),
        _ext_edge("D", "increases", "A"),
        _ext_edge("D", "increases", "B"),
        # C is a sink (no out-edges, in-edges from {A, B}); D is a
        # source (no in-edges, out-edges into {A, B}). C ≁ D because
        # their direction-tagged property sets differ.
    ]
    rng = np.random.default_rng(0)
    rep_of = structural_equivalence(propositions, edges, rng)

    # A and B share the representative; C and D are themselves.
    assert rep_of["A"] == rep_of["B"]
    assert rep_of["C"] == "C"
    assert rep_of["D"] == "D"
    # Three distinct representatives: {rep(A)=rep(B), C, D}.
    assert len(set(rep_of.values())) == 3


# ---------------------------------------------------------------------
# Test 2: idempotence — applying the structural quotient twice
# (re-running it on the *output* propositions) is a no-op.
# ---------------------------------------------------------------------

def test_idempotence():
    propositions = ["A", "B", "C"]
    edges = [
        _ext_edge("A", "causes", "C"),
        _ext_edge("B", "causes", "C"),
    ]
    rng = np.random.default_rng(0)
    rep_of_first = structural_equivalence(propositions, edges, rng)

    # Project the proposition set and edge set under the first quotient,
    # then apply the quotient again. Representatives must be unchanged.
    reps = sorted(set(rep_of_first.values()))
    edges_quot = [
        Edge(
            subj=rep_of_first[e["subj"]],
            rel=e["rel"],
            obj=rep_of_first[e["obj"]],
            domain=e["domain"],
            edge_source=e["edge_source"],
            provenance=e["provenance"],
        )
        for e in edges
    ]
    rng2 = np.random.default_rng(0)
    rep_of_second = structural_equivalence(reps, edges_quot, rng2)

    for p in reps:
        assert rep_of_second[p] == p, "second pass must not re-merge"


# ---------------------------------------------------------------------
# Test 3: representative selection prefers the longest member string;
# ties are broken deterministically given the seed.
# ---------------------------------------------------------------------

def test_representative_is_longest_string():
    # All three are indiscernible (no edges at all → empty property
    # set). The longest string wins.
    propositions = ["short", "medium-len", "the longest"]
    edges: list[Edge] = []
    rng = np.random.default_rng(0)
    rep_of = structural_equivalence(propositions, edges, rng)
    assert rep_of["short"] == "the longest"
    assert rep_of["medium-len"] == "the longest"
    assert rep_of["the longest"] == "the longest"


def test_representative_tie_broken_deterministically():
    # Two strings of equal length, both indiscernible. The chosen
    # representative must depend only on the seed.
    propositions = ["alpha", "bravo"]
    edges: list[Edge] = []
    rng_a = np.random.default_rng(42)
    rep_of_a = structural_equivalence(propositions, edges, rng_a)
    rng_b = np.random.default_rng(42)
    rep_of_b = structural_equivalence(propositions, edges, rng_b)
    assert rep_of_a == rep_of_b


# ---------------------------------------------------------------------
# Test 4: normalization. Two propositions differing only in case or
# surrounding whitespace are treated as the same node by the
# equivalence relation.
# ---------------------------------------------------------------------

def test_case_and_whitespace_normalized():
    # "A" and "  a  " are the same normalized node and so collapse
    # trivially. There is only one equivalence class.
    propositions = ["A", "  a  "]
    edges = [_ext_edge("A", "causes", "C")]
    rng = np.random.default_rng(0)
    rep_of = structural_equivalence(propositions, edges, rng)
    # Both surface strings map to the SAME representative.
    assert rep_of["A"] == rep_of["  a  "]


# ---------------------------------------------------------------------
# Test 5: schema conformance — the output records carry every field
# documented in the project description.
# ---------------------------------------------------------------------

def test_output_schema_conformance():
    extracted = [
        _ext_edge("alpha proposition", "causes", "beta outcome"),
        _ext_edge("alpha proposition", "causes", "beta outcome"),  # dup
    ]
    expansion = [
        _exp_edge("alpha proposition", "leads_to", "beta outcome"),
    ]
    rng = np.random.default_rng(0)
    rep_of = structural_equivalence(
        ["alpha proposition", "beta outcome"],
        extracted + expansion,
        rng,
    )
    records = _build_output_records(
        extracted=extracted,
        expansion=expansion,
        rep_of=rep_of,
        provenance_truncate=5,
    )

    assert records, "must produce at least one record"
    expected_fields = {
        "subj", "rel", "obj", "domain", "edge_source",
        "group_id_subj", "group_id_obj",
        "members_subj", "members_obj",
        "provenance", "provenance_truncated", "provenance_total_count",
    }
    for record in records:
        assert expected_fields.issubset(record.keys())
        assert isinstance(record["edge_source"], list)
        assert isinstance(record["members_subj"], list)
        assert isinstance(record["members_obj"], list)
        assert isinstance(record["provenance"], list)
        assert isinstance(record["provenance_truncated"], bool)
        assert isinstance(record["provenance_total_count"], int)

    # The (alpha, causes, beta) record reports only "extraction"; the
    # (alpha, leads_to, beta) record reports only "expansion".
    by_rel = {r["rel"]: r for r in records}
    assert by_rel["causes"]["edge_source"] == ["extraction"]
    assert by_rel["leads_to"]["edge_source"] == ["expansion"]


# ---------------------------------------------------------------------
# Test 6: provenance truncation respects the configured cap.
# ---------------------------------------------------------------------

def test_provenance_truncation():
    # Build 10 extraction edges all collapsing to the same (subj, rel,
    # obj). Each carries a unique provenance entry.
    extracted = []
    for i in range(10):
        e = _ext_edge("p", "causes", "q")
        e["provenance"] = [{"topic": f"t{i}", "path": [], "question": "",
                            "statement": ""}]
        extracted.append(e)

    rng = np.random.default_rng(0)
    rep_of = structural_equivalence(["p", "q"], extracted, rng)
    records = _build_output_records(
        extracted=extracted,
        expansion=[],
        rep_of=rep_of,
        provenance_truncate=5,
    )
    assert len(records) == 1
    rec = records[0]
    assert len(rec["provenance"]) == 5
    assert rec["provenance_truncated"] is True
    assert rec["provenance_total_count"] == 10


# ---------------------------------------------------------------------
# Test 7: edge-set monotonicity in quotient_triples — applying the
# end-to-end pass with a no-op LLM (returns unparseable response and
# a singleton fallback partition; per-class expansion produces no new
# edges because every class has size 1) leaves the underlying property
# sets unchanged. Each input proposition stays its own equivalence
# class.
# ---------------------------------------------------------------------

class _NoOpLLM:
    """An LLM whose responses always fail to parse, exercising the
    fallback paths: Step A returns singleton partitions; Step B drops
    the call (every class has size 1, so `expand_edges_for_class`
    returns empty without making a call anyway)."""

    def ask(self, prompt: str) -> str:
        return "not-a-valid-json"

    def ask_batch(self, prompts):
        return [self.ask(p) for p in prompts]


def _stub_cache():
    # Cache with dir=None → every get returns None, every put is a drop.
    from scripts._llm_cache import Cache
    return Cache(dir=None, model="m", temperature=0.0)


# ---------------------------------------------------------------------
# Relation equivalence (relation softening)
# ---------------------------------------------------------------------

def test_relation_equivalence_collapses_synonymous_verbs():
    """Two propositions with identical neighbours but different
    causal verbs (causes vs. leads_to) should NOT collapse under raw
    verb identity, but SHOULD collapse when both verbs map to the
    same canonical class."""
    propositions = ["P", "Q"]
    edges = [
        _ext_edge("P", "causes",   "X"),
        _ext_edge("Q", "leads_to", "X"),
    ]
    rng = np.random.default_rng(0)

    # No softening: P and Q have different property sets.
    rep_of_raw = structural_equivalence(propositions, edges, rng)
    assert rep_of_raw["P"] != rep_of_raw["Q"]

    # With Option-2-style softening: both verbs are PROMOTES, so the
    # property sets coincide and the two collapse.
    eq_map = {"causes": "PROMOTES", "leads_to": "PROMOTES"}
    rng2 = np.random.default_rng(0)
    rep_of_soft = structural_equivalence(
        propositions, edges, rng2, relation_equivalence=eq_map,
    )
    assert rep_of_soft["P"] == rep_of_soft["Q"]


def test_relation_equivalence_in_output_records():
    """Output records should report the canonical class as `rel` and
    the original verbs in `original_rels`."""
    extracted = [
        _ext_edge("p", "causes",   "q"),
        _ext_edge("p", "leads_to", "q"),
    ]
    rng = np.random.default_rng(0)
    eq_map = {"causes": "PROMOTES", "leads_to": "PROMOTES"}
    rep_of = structural_equivalence(
        ["p", "q"], extracted, rng, relation_equivalence=eq_map,
    )
    records = _build_output_records(
        extracted=extracted,
        expansion=[],
        rep_of=rep_of,
        provenance_truncate=5,
        relation_equivalence=eq_map,
    )
    # The two raw edges collapse to a single output record because
    # their relations share a canonical class.
    assert len(records) == 1
    rec = records[0]
    assert rec["rel"] == "PROMOTES"
    assert rec["original_rels"] == ["causes", "leads_to"]


# ---------------------------------------------------------------------
# Corroboration filter
# ---------------------------------------------------------------------

def test_corroboration_filter_drops_solo_assertions_and_keeps_agreed_cells():
    """Two members in a candidate set: P1 and P2.

      Extracted edges: (none, simulating Step B-only evidence).
      Expansion edges (LLM said "yes" for):
        - (P1, causes, X)  — P2 also asserts; corroborated, keep both
        - (P2, causes, X)  — corroborated, keep both
        - (P1, causes, Y)  — solo (P2 silent on this cell), drop
        - (P2, causes, Z)  — solo (P1 silent on this cell), drop
    """
    members = ["P1", "P2"]
    expansion = [
        _exp_edge("P1", "causes", "X"),
        _exp_edge("P2", "causes", "X"),
        _exp_edge("P1", "causes", "Y"),
        _exp_edge("P2", "causes", "Z"),
    ]
    expansion_per_class = [(members, expansion)]

    kept, n_dropped = _filter_uncorroborated_expansion_edges(
        expansion_per_class, extracted=[], relation_equivalence=None,
    )
    kept_keys = {(e["subj"], e["rel"], e["obj"]) for e in kept}
    assert kept_keys == {("P1", "causes", "X"), ("P2", "causes", "X")}
    assert n_dropped == 2  # the (..., causes, Y) and (..., causes, Z) edges


def test_corroboration_filter_uses_canonical_relation():
    """Corroboration is checked at the relation-equivalence-class
    level: if P1 has (causes, X) and P2 has (leads_to, X), and both
    verbs are PROMOTES, the cells corroborate one another."""
    members = ["P1", "P2"]
    expansion = [
        _exp_edge("P1", "causes",   "X"),
        _exp_edge("P2", "leads_to", "X"),
    ]
    eq_map = {"causes": "PROMOTES", "leads_to": "PROMOTES"}
    kept, n_dropped = _filter_uncorroborated_expansion_edges(
        [(members, expansion)],
        extracted=[],
        relation_equivalence=eq_map,
    )
    kept_keys = {(e["subj"], e["rel"], e["obj"]) for e in kept}
    assert kept_keys == {("P1", "causes", "X"), ("P2", "leads_to", "X")}
    assert n_dropped == 0


def test_corroboration_filter_extraction_provides_corroboration():
    """An extraction edge can corroborate an LLM-asserted expansion
    edge — extraction edges count toward the per-cell tally."""
    members = ["P1", "P2"]
    extracted = [_ext_edge("P1", "causes", "X")]
    expansion = [_exp_edge("P2", "causes", "X")]
    kept, n_dropped = _filter_uncorroborated_expansion_edges(
        [(members, expansion)],
        extracted=extracted,
        relation_equivalence=None,
    )
    assert len(kept) == 1
    assert (kept[0]["subj"], kept[0]["rel"], kept[0]["obj"]) == ("P2", "causes", "X")
    assert n_dropped == 0


def test_no_op_llm_yields_no_collapse():
    """End-to-end with a no-op LLM. The triples are constructed so
    that no two propositions are structurally indiscernible from the
    extracted edges alone — alpha and gamma point to *different*
    objects, so prop(alpha) ≠ prop(gamma). With Step A falling back to
    singletons and Step B contributing no expansion edges, the
    quotient leaves the graph entirely intact."""
    triples = [
        {"subj": "alpha", "rel": "causes", "obj": "beta1", "domain": "d",
         "topic": "d", "path": ["d"], "question": "q1", "statement": "s1"},
        {"subj": "gamma", "rel": "causes", "obj": "beta2", "domain": "d",
         "topic": "d", "path": ["d"], "question": "q2", "statement": "s2"},
    ]
    config = {
        "shard_size": 100,
        "max_class_size": 5,
        "max_probes_per_class": 10,
        "provenance_truncate": 5,
        "relations": ["causes"],
    }
    rng = np.random.default_rng(0)
    records, stats = quotient_triples(
        triples=triples,
        llm=_NoOpLLM(),
        cache=_stub_cache(),
        rng=rng,
        config=config,
    )
    # |V| = 4 (alpha, beta1, gamma, beta2). All distinct property sets,
    # so |V/~| = 4. No expansion edges (singletons skip Step B).
    assert stats["n_propositions"] == 4
    assert stats["n_classes"] == 4
    assert stats["n_expansion"] == 0
    assert len(records) == 2  # two distinct edges in G/~


# =====================================================================
# _triples_to_edges — input handling for both Module-4 raw triples
# and prior-iteration quotient-output records
# =====================================================================
#
# These tests pin the raw-verb-preservation contract. Without it,
# `relation_equivalence` ceases to be a Step-C-only coarsening when
# `max_iterations > 1` — the canonical class label silently overwrites
# the verb on the way back into Step B / Step C, and verb-level
# semantics are lost from iteration 2 onward.

def test_triples_to_edges_module4_raw_triple_uses_rel_field():
    """Module-4-style triples have no `original_rels`; `rel` is the
    raw verb and we get exactly one edge per row."""
    triples = [
        {"subj": "A", "rel": "causes", "obj": "B",
         "topic": "t1", "path": ["t1"], "question": "q?",
         "statement": "A causes B."},
    ]
    edges = _triples_to_edges(triples)
    assert len(edges) == 1
    e = edges[0]
    assert e["subj"] == "A"
    assert e["rel"]  == "causes"
    assert e["obj"]  == "B"
    assert e["edge_source"] == "extraction"
    # Provenance reconstructed from top-level Module-4 fields.
    assert len(e["provenance"]) == 1
    assert e["provenance"][0]["statement"] == "A causes B."


def test_triples_to_edges_quotient_record_expands_per_original_rel():
    """A quotient-output record (with `original_rels`) expands to ONE
    edge per raw verb, each preserving the verb in `rel`. This is the
    core fix: iteration must NOT collapse verbs into the canonical
    class label between iterations."""
    record = {
        "subj":          "A",
        "rel":           "ASSOCIATES",            # canonical class label
        "obj":           "B",
        "domain":        "d1",
        "edge_source":   ["extraction"],
        "original_rels": ["causes", "leads_to"],  # raw verbs preserved here
        "group_id_subj": "g_000",
        "group_id_obj":  "g_001",
        "members_subj":  ["A"],
        "members_obj":   ["B"],
        "provenance":    [{"topic": "t", "path": ["t"], "question": "q?",
                           "statement": "A causes B."}],
        "provenance_truncated": False,
        "provenance_total_count": 1,
    }
    edges = _triples_to_edges([record])
    assert len(edges) == 2, (
        "expected one edge per raw verb in original_rels; the canonical "
        "class label in `rel` must NOT be used"
    )
    rels = sorted(e["rel"] for e in edges)
    assert rels == ["causes", "leads_to"]
    for e in edges:
        assert e["subj"] == "A"
        assert e["obj"]  == "B"
        assert e["domain"] == "d1"
        # Provenance carried through verbatim, shared across split edges.
        assert e["provenance"] == record["provenance"]


def test_triples_to_edges_round_trip_preserves_raw_verbs():
    """Build a Step-C output via _build_output_records (which writes
    the canonical rel + original_rels), then feed it back through
    _triples_to_edges and confirm the edge list has the raw verbs, not
    the canonical class label."""
    extracted = [
        _ext_edge("A", "causes", "B"),
        _ext_edge("A", "leads_to", "B"),
    ]
    rep_of = {"A": "A", "B": "B"}
    relation_equivalence = {"causes": "ASSOCIATES", "leads_to": "ASSOCIATES"}
    records = _build_output_records(
        extracted=extracted,
        expansion=[],
        rep_of=rep_of,
        provenance_truncate=10,
        relation_equivalence=relation_equivalence,
    )
    # The single output record's `rel` is the canonical class.
    assert len(records) == 1
    assert records[0]["rel"] == "ASSOCIATES"
    assert sorted(records[0]["original_rels"]) == ["causes", "leads_to"]

    # Feeding back must recover the raw verbs, NOT the canonical label.
    edges = _triples_to_edges(records)
    rels = sorted(e["rel"] for e in edges)
    assert rels == ["causes", "leads_to"], (
        "round-trip lost raw verb info; iteration would erase verbs "
        "from iter 2 onward"
    )
    assert "ASSOCIATES" not in rels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

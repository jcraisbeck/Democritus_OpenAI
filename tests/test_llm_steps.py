"""Tests for the LLM-touching portions of the indiscernibility
quotient: Step A (partition_propositions) and Step B
(expand_edges_for_class).

A FakeLLM class supplies canned responses so the tests are offline,
deterministic, and fast. Each test names the behaviour it exercises:

  - happy path
  - markdown-fence stripping
  - malformed-output retry
  - malformed-output exhaustion (singleton fallback for A;
    drop for B)
  - hallucination guard (out-of-vocab relation or node → filtered)
  - per-class enumeration: ONE LLM call per class, only true
    assertions returned

Run with:

    pytest Democritus_OpenAI/tests/test_llm_steps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest


_HERE = Path(__file__).resolve()
_DEMOCRITUS_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_DEMOCRITUS_ROOT))

from scripts._llm_cache import Cache                                  # noqa: E402
from scripts.indiscernibility_quotient import (                       # noqa: E402
    Edge,
    expand_edges_for_class,
    partition_propositions,
)


# ---------------------------------------------------------------------
# FakeLLM: returns scripted responses keyed by prompt-substring match.
# Counts how many times `ask` is invoked so we can assert one-call-per-
# class for Step B and one-call-per-shard for Step A.
# ---------------------------------------------------------------------

class FakeLLM:
    """An LLM that returns a scripted response for the first prompt
    whose substring rule matches. If no rule matches, returns the
    `default` string (use for forcing the fallback path).

    Rules are a list of (substring, response) pairs evaluated in
    order; the first match wins.
    """

    def __init__(self, rules: List[tuple], default: str = "not-json"):
        self.rules = rules
        self.default = default
        self.calls: List[str] = []

    def ask(self, prompt: str) -> str:
        self.calls.append(prompt)
        for substring, response in self.rules:
            if substring in prompt:
                return response
        return self.default

    def ask_batch(self, prompts):
        return [self.ask(p) for p in prompts]


def _no_cache() -> Cache:
    return Cache(dir=None, model="m", temperature=0.0)


# =====================================================================
# Step A — partition_propositions (now: candidate similarity sets)
# =====================================================================
#
# Step A no longer returns a partition; it returns a list of CONFIDENT
# similarity sets. Most propositions belong to no set. Indices may
# overlap across sets. Sets of size < 2 are dropped (no comparative
# information).

# Test A1: happy path. The LLM returns one confident set; the function
# returns one candidate class.

def test_step_a_happy_path():
    propositions = ["alpha", "beta", "gamma"]
    # Sorted: ["alpha", "beta", "gamma"]. The LLM thinks alpha and
    # beta are near-paraphrases; gamma stands alone (no set).
    response = json.dumps({"sets": [[0, 1]]})
    llm = FakeLLM(rules=[("alpha", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    assert classes == [["alpha", "beta"]], (
        "gamma must NOT be added as a singleton — it is correctly "
        "absent from the LLM's output"
    )
    assert len(llm.calls) == 1, "exactly one LLM call for one shard"


# Test A2: markdown-fence stripping. The LLM wraps its JSON in a
# ```json ... ``` block; the parser must still accept it.

def test_step_a_strips_markdown_fences():
    propositions = ["x", "y", "z"]
    response = "```json\n" + json.dumps({"sets": [[0, 2]]}) + "\n```"
    llm = FakeLLM(rules=[("x", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    assert classes == [["x", "z"]]


# Test A3: malformed-output retry. The first response is unparseable;
# the second is valid. Two LLM calls are issued; the result is the set
# from the second response.

def test_step_a_retries_on_malformed_then_succeeds():
    propositions = ["a", "b"]
    good = json.dumps({"sets": [[0, 1]]})
    class _Toggle:
        def __init__(self) -> None:
            self.calls = 0

        def ask(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "garbage not json"
            return good

        def ask_batch(self, prompts):
            return [self.ask(p) for p in prompts]

    llm = _Toggle()
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    assert classes == [["a", "b"]]
    assert llm.calls == 2  # one initial + one retry


# Test A4: empty-but-parseable response is accepted as "no candidate
# sets" without retry. Step C still runs (and finds nothing
# interesting).

def test_step_a_empty_sets_is_valid_no_retry():
    propositions = ["x", "y", "z"]
    response = json.dumps({"sets": []})
    llm = FakeLLM(rules=[("x", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    assert classes == []
    assert len(llm.calls) == 1, "no retry on parseable empty response"


# Test A5: malformed-output exhaustion → no candidate sets for the
# shard. Two LLM calls are issued; the result is empty (Step B will
# do nothing; Step C still runs).

def test_step_a_exhausts_to_no_sets():
    propositions = ["x", "y", "z"]
    llm = FakeLLM(rules=[], default="garbage")
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    assert classes == []
    assert len(llm.calls) == 2  # initial + retry


# Test A6: oversized set is split. The LLM returns a single confident
# set with 5 members; max_class_size=2 → split into 3 cells
# (sizes 2, 2, 1). The size-1 cell will simply be skipped by Step B.

def test_step_a_splits_oversized_sets():
    propositions = ["a", "b", "c", "d", "e"]
    response = json.dumps({"sets": [[0, 1, 2, 3, 4]]})
    llm = FakeLLM(rules=[("a", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=2,
    )
    sizes = sorted(len(c) for c in classes)
    assert sizes == [1, 2, 2]


# Test A7: overlap across sets is allowed. The LLM puts index 0 in
# both sets — this is intentional, supporting propositions that are
# near-paraphrases of multiple distinct claims.

def test_step_a_allows_overlapping_sets():
    propositions = ["a", "b", "c"]
    response = json.dumps({"sets": [[0, 1], [0, 2]]})
    llm = FakeLLM(rules=[("a", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    # Both sets are kept; "a" appears in both.
    assert sorted([sorted(c) for c in classes]) == [["a", "b"], ["a", "c"]]


# Test A8: singleton sets returned by the LLM are dropped (they carry
# no comparative information for Step B).

def test_step_a_drops_singleton_sets():
    propositions = ["a", "b", "c"]
    # Two sets: a singleton {0} (dropped) and a real pair {1, 2}.
    response = json.dumps({"sets": [[0], [1, 2]]})
    llm = FakeLLM(rules=[("a", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
    )
    assert classes == [["b", "c"]]


# =====================================================================
# Step B — expand_edges_for_class
# =====================================================================

def _ext(s: str, r: str, o: str) -> Edge:
    return Edge(subj=s, rel=r, obj=o, domain="d0",
                edge_source="extraction", provenance=[])


# Test B1: per-cell enumeration. ONE LLM call per class. The LLM
# enumerates true cells per (member, relation, probe, direction); the
# function emits one Edge per asserted cell that is not already in the
# extracted set. NO propagation across members — each member gets the
# LLM's independent judgment.

def test_expansion_one_call_per_class_returns_new_edges_only():
    members = ["alpha", "beta"]
    edges = [
        _ext("alpha", "causes", "gamma"),
        _ext("beta",  "causes", "delta"),
    ]
    relations = ["causes", "leads_to"]

    # Probes after sorting by degree+alphabetical: ["delta", "gamma"]
    # (both degree 1; alphabetical tiebreak). z=0 is delta, z=1 is gamma.
    response = json.dumps({"asserted": [
        {"m": 0, "r": "causes",   "z": 1, "d": "out"},   # already extracted
        {"m": 1, "r": "leads_to", "z": 1, "d": "out"},   # new
        {"m": 0, "r": "causes",   "z": 1, "d": "in"},    # new (gamma->alpha)
    ]})
    llm = FakeLLM(rules=[("alpha", response)])

    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="exhaustive",
    )
    assert len(llm.calls) == 1, "Step B: ONE LLM call per class"
    assert dropped == 0
    keys = {(e["subj"], e["rel"], e["obj"]) for e in new_edges}
    assert ("beta",  "leads_to", "gamma") in keys
    assert ("gamma", "causes",   "alpha") in keys
    assert len(new_edges) == 2
    assert all(e["edge_source"] == "expansion" for e in new_edges)


# Test B2: hallucination guard. The LLM returns cells with out-of-
# vocabulary relations and out-of-range indices; all such cells are
# dropped and counted.

def test_expansion_hallucination_guard():
    members = ["a", "b"]
    edges = [_ext("a", "causes", "x"), _ext("b", "causes", "y")]
    relations = ["causes"]
    response = json.dumps({"asserted": [
        {"m": 0, "r": "causes",  "z": 0, "d": "out"},      # valid; duplicates extracted
        {"m": 0, "r": "FAKEREL", "z": 0, "d": "out"},      # bad relation
        {"m": 5, "r": "causes",  "z": 0, "d": "out"},      # bad m index
        {"m": 0, "r": "causes",  "z": 99, "d": "out"},     # bad z index
        {"m": 0, "r": "causes",  "z": 0, "d": "sideways"}, # bad direction
    ]})
    llm = FakeLLM(rules=[("a", response)])
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="exhaustive",
    )
    # Valid cell duplicates an extracted edge; four bad cells dropped.
    assert dropped == 4
    assert new_edges == []


# Test B3: singleton class — no LLM call is issued because there is
# nothing to compare. Returns ([], 0).

def test_expansion_singleton_class_skips_llm():
    llm = FakeLLM(rules=[], default="garbage")  # would fail if called
    new_edges, dropped = expand_edges_for_class(
        members=["alone"],
        member_contexts=[None],
        edges=[_ext("alone", "causes", "x")],
        relations=["causes"],
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="exhaustive",
    )
    assert new_edges == []
    assert dropped == 0
    assert llm.calls == [], "no LLM call should be made for singleton class"


# Test B4: empty 1-hop neighbourhood (members exist but have no
# extracted edges) — no probe set, no LLM call.

def test_expansion_no_neighbours_skips_llm():
    members = ["p", "q"]
    edges: list[Edge] = []  # no extracted edges
    llm = FakeLLM(rules=[], default="garbage")
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=["causes"],
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="exhaustive",
    )
    assert new_edges == []
    assert dropped == 0
    assert llm.calls == []


# Test B5: probe truncation. With a class connected to 100 distinct
# probes and max_probes=3, only the top-3-by-degree probes appear in
# the LLM prompt. The function still issues exactly one call.

def test_expansion_truncates_probes_by_degree():
    members = ["m"]
    # Build edges: m has 100 distinct neighbours via "causes". Add one
    # extra edge into one of the neighbours so it has higher degree.
    edges: list[Edge] = []
    members.append("m2")  # need ≥ 2 members to trigger an LLM call
    # m connected to z0..z99
    for i in range(100):
        edges.append(_ext("m", "causes", f"z{i}"))
    # boost z42's degree by adding another edge into it
    edges.append(_ext("m2", "causes", "z42"))

    response = json.dumps({"asserted": []})
    # The probe truncation puts "z42" first (highest degree); using it
    # as the rule trigger guarantees the rule matches the constructed
    # prompt regardless of spacing details.
    llm = FakeLLM(rules=[("z42", response)])
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=["causes"],
        max_probes=3,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="exhaustive",
    )
    assert len(llm.calls) == 1
    prompt = llm.calls[0]
    # z42 must appear (highest-degree probe). We do not assert which
    # other two appear — alphabetical tie-break among degree-1 probes.
    assert "z42" in prompt


# Test B6: skip-already-connected. With the flag on, an LLM-asserted
# cell is dropped if its (subj, obj) pair is already connected by an
# extraction edge — regardless of direction or relation. Only cells
# whose pairs are unconnected survive.

def test_expansion_skips_already_connected_pairs():
    members = ["P1", "P2"]
    edges = [
        # Two extraction edges so the 1-hop neighbourhood contains both
        # X (neighbour of P1) and Z (neighbour of P2). Pair {P1, X}
        # and pair {P2, Z} are each already connected by extraction;
        # the cross-pair {P1, Z} is NOT.
        _ext("P1", "causes", "X"),
        _ext("P2", "causes", "Z"),
    ]
    relations = ["causes", "leads_to"]
    # Probes: 1-hop neighbours of {P1, P2} excluding members =
    # {X, Z}. Both degree 1; alphabetical order gives ["X", "Z"].
    # So z=0 is X, z=1 is Z.
    response = json.dumps({"asserted": [
        {"m": 0, "r": "leads_to", "z": 0, "d": "out"},   # (P1, leads_to, X) — pair {P1,X} already connected → SKIP
        {"m": 1, "r": "leads_to", "z": 1, "d": "out"},   # (P2, leads_to, Z) — pair {P2,Z} already connected → SKIP
        {"m": 0, "r": "leads_to", "z": 1, "d": "out"},   # (P1, leads_to, Z) — pair {P1,Z} unconnected → KEEP
    ]})
    llm = FakeLLM(rules=[("P1", response)])
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=True,
        mode="exhaustive",
    )
    # Probes are stored as normalized (lowercase) strings, so the obj
    # of the new edge is "z" (not "Z"). Members keep their original case.
    keys = {(e["subj"], e["rel"], e["obj"]) for e in new_edges}
    assert keys == {("P1", "leads_to", "z")}


# Test B7: undirected check. If extraction has (X causes P1), then
# (P1, leads_to, X) — same pair, opposite direction — is also skipped.

def test_expansion_skip_already_connected_is_undirected():
    members = ["P1", "P2"]
    edges = [_ext("X", "causes", "P1")]    # extraction: X -> P1
    relations = ["leads_to"]
    # Probes: just X (1-hop neighbour of P1). z=0 is X.
    response = json.dumps({"asserted": [
        {"m": 0, "r": "leads_to", "z": 0, "d": "out"},   # P1 -> X, opposite direction; SKIP
    ]})
    llm = FakeLLM(rules=[("P1", response)])
    new_edges, _ = expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=True,
        mode="exhaustive",
    )
    assert new_edges == [], "undirected pair check should suppress reverse-direction expansion"


# =====================================================================
# Step A — pivot-mode tests (snippet grounding + rep designation)
# =====================================================================

# Marker substrings to disambiguate the two pivot Step B prompts in
# FakeLLM rules.
_PHASE1_MARKER = "currently asserted on each NON-REP"
_PHASE2_MARKER = "each causal edge currently asserted on the REP"


# Test PA1: pivot-mode snippet validation. The LLM emits one set with
# three members; one member's snippet does not match the proposition at
# its index. That member is dropped, leaving a valid 2-member set.

def test_step_a_pivot_snippet_validation_drops_drift():
    propositions = ["alpha is a proposition", "beta is too", "gamma stands alone"]
    response = json.dumps({"sets": [
        [
            {"i": 0, "snippet": "alpha"},
            {"i": 1, "snippet": "beta"},
            {"i": 2, "snippet": "WRONGWORDS"},
        ],
    ]})
    llm = FakeLLM(rules=[("alpha is a proposition", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
        step_b_mode="pivot",
    )
    # After validation the set has alpha, beta. Order preserved → rep is
    # "alpha is a proposition" (first), then "beta is too".
    assert classes == [["alpha is a proposition", "beta is too"]]


# Test PA2: pivot-mode rep designation. The LLM puts member at index 1
# first (designating it as rep) and the member at index 0 second. The
# returned class preserves that ordering.

def test_step_a_pivot_rep_is_first_returned_member():
    propositions = ["claim X", "claim Y", "claim Z"]
    response = json.dumps({"sets": [
        [
            {"i": 1, "snippet": "claim Y"},   # rep
            {"i": 0, "snippet": "claim X"},
        ],
    ]})
    llm = FakeLLM(rules=[("claim X", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
        step_b_mode="pivot",
    )
    assert classes == [["claim Y", "claim X"]]


# Test PA3: a set whose REP itself fails snippet validation is dropped
# entirely (no anchor for downstream Phase 1/2).

def test_step_a_pivot_drops_set_when_rep_snippet_invalid():
    propositions = ["alpha is a proposition", "beta is too"]
    response = json.dumps({"sets": [
        [
            {"i": 0, "snippet": "WRONGWORDS"},  # rep snippet fails → drop set
            {"i": 1, "snippet": "beta"},
        ],
    ]})
    llm = FakeLLM(rules=[("alpha is a proposition", response)])
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=_no_cache(),
        shard_size=100,
        max_class_size=10,
        step_b_mode="pivot",
    )
    assert classes == []


# =====================================================================
# Step B — pivot-mode tests
# =====================================================================

# Test PB1: Phase 1 happy path. Non-rep member has one extraction edge;
# the LLM asserts that the analogous edge holds for the rep too. Result
# is a new expansion edge anchored on the rep.

def test_pivot_phase1_lifts_member_edges_to_rep():
    members = ["rep-claim", "other-claim"]                 # rep first
    edges = [_ext("other-claim", "causes", "z1")]          # non-rep has 1 edge
    relations = ["causes"]

    phase1 = json.dumps({"asserted": [
        {"m": 0, "e": 0, "holds": True},                   # lift to rep
    ]})
    # Phase 2's input rep_edges = [(out, causes, z1)]; LLM agrees, but
    # the resulting (other-claim, causes, z1) duplicates the existing
    # extraction edge so the emit is suppressed.
    phase2 = json.dumps({"asserted": [
        {"m": 0, "e": 0, "holds": True},
    ]})
    llm = FakeLLM(rules=[
        (_PHASE1_MARKER, phase1),
        (_PHASE2_MARKER, phase2),
    ])
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None, None],
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="pivot",
    )
    assert dropped == 0
    assert len(llm.calls) == 2, "pivot mode: Phase 1 + Phase 2"
    keys = {(e["subj"], e["rel"], e["obj"]) for e in new_edges}
    assert ("rep-claim", "causes", "z1") in keys
    assert all(e["edge_source"] == "expansion" for e in new_edges)


# Test PB2: Phase 2 happy path. Rep has an extraction edge; non-rep has
# none → Phase 1 is skipped. Phase 2 distributes the rep's edge onto
# the non-rep when the LLM says yes.

def test_pivot_phase2_distributes_rep_edges_to_others():
    members = ["rep-claim", "other-claim"]
    edges = [_ext("rep-claim", "causes", "z1")]
    relations = ["causes"]

    phase2 = json.dumps({"asserted": [
        {"m": 0, "e": 0, "holds": True},
    ]})
    llm = FakeLLM(rules=[(_PHASE2_MARKER, phase2)])
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None, None],
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="pivot",
    )
    assert dropped == 0
    assert len(llm.calls) == 1, "Phase 1 skipped (no member edges to lift)"
    keys = {(e["subj"], e["rel"], e["obj"]) for e in new_edges}
    assert ("other-claim", "causes", "z1") in keys
    assert all(e["edge_source"] == "expansion" for e in new_edges)


# Test PB3: end-to-end call count. Both rep and non-rep members have
# extraction edges so both phases fire. Exactly two LLM calls per
# class regardless of class size.

def test_pivot_two_calls_per_class():
    members = ["rep-claim", "M1", "M2", "M3"]
    edges = [
        _ext("rep-claim", "causes", "z0"),
        _ext("M1", "causes", "z1"),
        _ext("M2", "causes", "z2"),
        _ext("M3", "causes", "z3"),
    ]
    relations = ["causes"]
    empty = json.dumps({"asserted": []})
    llm = FakeLLM(rules=[
        (_PHASE1_MARKER, empty),
        (_PHASE2_MARKER, empty),
    ])
    expand_edges_for_class(
        members=members,
        member_contexts=[None] * len(members),
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="pivot",
    )
    assert len(llm.calls) == 2, (
        "pivot mode issues exactly 2 LLM calls per class (Phase 1 + Phase 2), "
        "regardless of class size"
    )


# Test PB4: hallucination guard. Phase 1 returns a mix of valid and
# invalid cells (bad indices, missing keys, non-bool holds). Bad cells
# are dropped and counted; the one valid cell still fires.

def test_pivot_hallucination_guard():
    members = ["rep", "other"]
    edges = [_ext("other", "causes", "z1")]
    relations = ["causes"]

    phase1 = json.dumps({"asserted": [
        {"m": 0, "e": 0, "holds": True},          # valid
        {"m": 0, "e": 99, "holds": True},         # bad e → drop
        {"m": 5, "e": 0, "holds": True},          # bad m → drop
        {"m": 0, "e": 0, "holds": "yes"},         # non-bool holds → drop
        {"missing": "keys"},                       # missing m/e/holds → drop
    ]})
    phase2 = json.dumps({"asserted": []})
    llm = FakeLLM(rules=[
        (_PHASE1_MARKER, phase1),
        (_PHASE2_MARKER, phase2),
    ])
    new_edges, dropped = expand_edges_for_class(
        members=members,
        member_contexts=[None, None],
        edges=edges,
        relations=relations,
        max_probes=10,
        llm=llm,
        cache=_no_cache(),
        skip_already_connected_pairs=False,
        mode="pivot",
    )
    # Note: the "non-bool holds" duplicate (m=0,e=0) is caught by the
    # holds-type check BEFORE dedup, so it counts as a hallucination
    # (the first valid cell uses the (0,0) key after passing all checks).
    assert dropped == 4
    keys = {(e["subj"], e["rel"], e["obj"]) for e in new_edges}
    assert ("rep", "causes", "z1") in keys


# Test PB5: cache-stability. Two equivalent edge multisets in different
# input orders must produce identical (direction, rel, neighbour) lists
# from `_member_edges_for_prompt`, so the formatted prompt is stable.

def test_pivot_member_edges_deterministic_ordering():
    from scripts.indiscernibility_quotient import _member_edges_for_prompt
    edges_a = [
        _ext("M", "causes", "Y"),
        _ext("M", "causes", "X"),
        _ext("Z", "leads_to", "M"),
    ]
    edges_b = [
        _ext("Z", "leads_to", "M"),
        _ext("M", "causes", "X"),
        _ext("M", "causes", "Y"),
    ]
    out_a = _member_edges_for_prompt("M", edges_a)
    out_b = _member_edges_for_prompt("M", edges_b)
    assert out_a == out_b
    # Sorted by (direction, rel, normalised neighbour). "in" < "out"
    # alphabetically, so the (in, leads_to, Z) tuple comes first.
    assert out_a[0] == ("in", "leads_to", "Z")
    assert out_a[1] == ("out", "causes", "X")
    assert out_a[2] == ("out", "causes", "Y")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

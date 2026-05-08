#!/usr/bin/env python3
"""
indiscernibility_quotient.py
----------------------------

Optional Module 4.5 — quotient the relational graph by Leibniz's
identity of indiscernibles.

Mechanism (three steps):

  Step A. Ask the LLM to identify CONFIDENT SETS OF NEAR-PARAPHRASES
          among the proposition set V. These sets do not partition V
          (most propositions belong to no set, and a proposition may
          appear in multiple sets). Precision-oriented: only items the
          LLM is confident about become candidate similarity classes.

  Step B. For each candidate set Cᵢ, gather the union of the 1-hop
          neighbourhoods of all members; one LLM call asks the LLM to
          enumerate every (member, relation, probe, direction) for
          which the causal claim is true. Asserted edges not already
          in E are added with edge_source = "expansion".

  Step C. Compute the equivalence relation
              P₁ ~ P₂  iff  prop(P₁) = prop(P₂)
          where prop(P) = {("out", r, z) : (P,r,z) ∈ E}
                       ∪  {("in",  r, z) : (z,r,P) ∈ E}.
          Quotient G by ~. The merger criterion is genuine equality of
          property sets — Leibniz's identity of indiscernibles applied
          to the relational data.

The module reads CONFIG from configs/quotient_config.py at entry and
operates on the current working directory. The pipeline orchestrator
chdirs into the run directory before invoking main().

Side effects (main):
    - Reads {input_filename}.
    - Writes {output_filename}.
    - Reads/writes the LLM disk cache under {cache_dir}.
    - Issues LLM network calls (one per Step A shard plus one per
      Step B class).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

from configs.quotient_config import CONFIG
from llms.base import LLMClient
from llms.factory import make_llm_client
from scripts._llm_cache import Cache, cached_ask


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

class Edge(TypedDict):
    subj: str
    rel: str
    obj: str
    domain: str
    # edge_source is named with the `edge_` prefix to avoid colliding
    # with manifold_builder._pick_key, which treats "source" as a
    # candidate key for the subject string. Without the prefix, our
    # list value crashes Module 5's hash lookup.
    edge_source: str        # "extraction" or "expansion"
    provenance: List[Dict[str, Any]]


# ---------------------------------------------------------------------
# Normalization
#
# All hashing of nodes and relations downstream assumes lowercase-stripped
# strings. Module 5's manifold_builder applies the same normalization
# implicitly when it constructs `vars`. Apply once at module entry.
# ---------------------------------------------------------------------

def _norm(s: str) -> str:
    return s.strip().lower()


# ---------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove a single surrounding markdown code fence, if present."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _safe_json_loads(text: str) -> Optional[Any]:
    """Try json.loads after stripping fences. Returns None on failure."""
    try:
        return json.loads(_strip_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Prompt access
#
# All LLM-facing strings are loaded from CONFIG["prompts"] (see
# configs/quotient_config.py). Builders below format the templates;
# nothing in this module hard-codes prompt text.
#
# `_EXPECTED_PLACEHOLDERS` records the format keys each builder will
# pass. The validator runs at module import time and fails loudly if
# a template's placeholders drift away from what the code provides
# (e.g. someone removes `{rel_list}` from a Phase 1 edit) — better to
# crash on import than to silently strip the relation vocabulary out
# of an LLM call.
# ---------------------------------------------------------------------

_PROMPTS: Dict[str, str] = CONFIG["prompts"]

_EXPECTED_PLACEHOLDERS: Dict[str, set] = {
    "step_a_exhaustive":         {"max_class_size", "numbered_list"},
    "step_a_pivot":              {"max_class_size", "numbered_list"},
    "step_b_exhaustive":         {"numbered_members_with_context",
                                  "numbered_probes", "rel_list"},
    "step_b_phase1_lift":        {"rep_block",
                                  "numbered_members_with_edges", "rel_list"},
    "step_b_phase2_distribute":  {"rep_block_with_edges",
                                  "numbered_members_with_context", "rel_list"},
    "retry_suffix":              {"prior_response"},
}

_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


def _validate_prompt_placeholders() -> None:
    """Confirm every active template's placeholders match what the
    builders will pass. Raises ValueError on mismatch.

    `step_a_legacy_v6` is documentation-only and not validated.
    """
    for key, expected in _EXPECTED_PLACEHOLDERS.items():
        if key not in _PROMPTS:
            raise ValueError(
                f"CONFIG['prompts'] missing required key {key!r}"
            )
        found = set(_PLACEHOLDER_RE.findall(_PROMPTS[key]))
        missing = expected - found
        extra = found - expected
        if missing or extra:
            raise ValueError(
                f"CONFIG['prompts'][{key!r}] placeholder mismatch: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )


_validate_prompt_placeholders()


def _format_prompt(key: str, **kwargs: Any) -> str:
    """Format the named prompt template with the given kwargs.

    Single chokepoint for prompt rendering — every LLM call in this
    module routes through here, so swapping a template (or replacing
    the dict with a different config source) requires changes only at
    the config layer."""
    return _PROMPTS[key].format(**kwargs)


def _retry_prompt(original_prompt: str, prior_response: str) -> str:
    """Build a one-shot retry prompt: original prompt + the unparseable
    response + an instruction to return only JSON.

    Used by every Step A / Step B call site that issues a JSON-expecting
    LLM call. Centralising the suffix here means the wording lives in
    a single place (CONFIG["prompts"]["retry_suffix"]) rather than being
    duplicated at four call sites."""
    return original_prompt + _format_prompt(
        "retry_suffix", prior_response=prior_response,
    )


# =====================================================================
# Step A — Candidate partition
# =====================================================================
#
# Two prompts exist in CONFIG["prompts"]: `step_a_exhaustive` (active
# under step_b_mode="exhaustive") and `step_a_pivot` (active under
# step_b_mode="pivot"). A third, `step_a_legacy_v6`, is preserved as
# documentation of the v6 precision-biased ancestor.
#
# Why we changed v6 → v7: the v6 conservative prompt produced 117
# candidate sets on |V| = 477, with 79% being pairs (average set size
# 2.67). Step C found 38 mergers (8% node compression). Inspection
# showed Step A was the bottleneck — it surfaced obvious paraphrases
# but missed cross-shard / cross-topic / mechanism-level near-claims.
# The conservative wording explicitly trained the LLM toward this:
#   "Precision matters; recall does not."
#   "Most propositions will not [be in any set]. That is correct."
#   "Stop emitting sets once no further confident groupings remain."
# All three sentences gave the LLM permission to be lazy.
#
# The new wording flips the bias: under-grouping is permanent (a missed
# pair never gets considered), over-grouping is cheap (Step C's strict
# Leibniz check filters out the false positives anyway). It also names
# five concrete kinds of similarity to look for rather than relying on
# the LLM's default "paraphrase" intuition. The expectation is more,
# and possibly larger, candidate sets — a higher Step B call budget,
# but more raw material for Step C to convert into mergers.


def _build_similarity_sets_prompt(
    propositions: List[str], max_class_size: int,
) -> str:
    numbered = "\n".join(f"{i}: {p}" for i, p in enumerate(propositions))
    return _format_prompt(
        "step_a_exhaustive",
        max_class_size=max_class_size,
        numbered_list=numbered,
    )


def _parse_similarity_sets_response(
    text: str,
    n_propositions: int,
) -> Optional[List[List[int]]]:
    """Parse the response into a list of candidate similarity sets.

    No coverage requirement: indices not in any set are left out (those
    propositions are not investigated by Step B). Each index may appear
    in multiple sets — overlap is permitted and useful, since a
    proposition may be a near-paraphrase of distinct claims.

    Filtering applied:
      - Out-of-range or non-integer indices are dropped.
      - Within a single set, duplicate indices are dropped.
      - Sets of size < 2 are dropped (they carry no comparative
        information for Step B).

    Returns:
      list of sets (possibly empty) if the JSON envelope is
        recoverable;
      None if the envelope is unrecoverable (no parseable JSON, or
        no `sets` key with a list value) — caller should retry once
        and then give up.
    """
    payload = _safe_json_loads(text)
    if not isinstance(payload, dict):
        return None
    sets = payload.get("sets")
    if not isinstance(sets, list):
        return None

    out: List[List[int]] = []
    for s in sets:
        if not isinstance(s, list):
            continue
        cell: List[int] = []
        seen_in_cell: set = set()
        for idx in s:
            if not isinstance(idx, int) or idx < 0 or idx >= n_propositions:
                continue
            if idx in seen_in_cell:
                continue
            seen_in_cell.add(idx)
            cell.append(idx)
        if len(cell) >= 2:
            out.append(cell)

    return out


def _split_oversized(
    classes: List[List[str]],
    max_class_size: int,
) -> List[List[str]]:
    """Deterministically split any class with more than `max_class_size`
    members into contiguous chunks of size `max_class_size`."""
    out: List[List[str]] = []
    for cell in classes:
        if len(cell) <= max_class_size:
            out.append(cell)
            continue
        for i in range(0, len(cell), max_class_size):
            out.append(cell[i : i + max_class_size])
    return out


def _split_oversized_pivot(
    classes: List[List[str]],
    max_class_size: int,
) -> List[List[str]]:
    """Pivot-mode variant of `_split_oversized`. When a class exceeds
    `max_class_size`, split into chunks of `max_class_size`. Every
    chunk after the first gets the original rep (`cell[0]`) prepended,
    so each chunk has the same LLM-designated representative.

    Chunks after the first thus carry one fewer fresh member than the
    first chunk (rep takes the leading slot)."""
    out: List[List[str]] = []
    for cell in classes:
        if len(cell) <= max_class_size:
            out.append(cell)
            continue
        out.append(cell[:max_class_size])
        rep = cell[0]
        step = max_class_size - 1
        for i in range(max_class_size, len(cell), step):
            chunk = [rep] + cell[i : i + step]
            out.append(chunk)
    return out


def _snippet_matches(snippet: str, proposition: str) -> bool:
    """Case-insensitive substring match with whitespace collapsed.

    Returns True iff the lowercased, whitespace-normalised snippet
    appears as a substring of the lowercased, whitespace-normalised
    proposition. Permissive by design: catches LLM index drift (the
    snippet bears no relation to the indexed proposition) without
    rejecting legitimate shortened renderings."""
    if not snippet or not proposition:
        return False
    norm_snippet = " ".join(snippet.lower().split())
    norm_prop = " ".join(proposition.lower().split())
    if not norm_snippet:
        return False
    return norm_snippet in norm_prop


def _build_similarity_sets_prompt_pivot(
    propositions: List[str], max_class_size: int,
) -> str:
    numbered = "\n".join(f"{i}: {p}" for i, p in enumerate(propositions))
    return _format_prompt(
        "step_a_pivot",
        max_class_size=max_class_size,
        numbered_list=numbered,
    )


def _parse_similarity_sets_response_pivot(
    text: str,
    propositions: List[str],
) -> Optional[List[List[int]]]:
    """Parse the pivot-mode similarity-sets response.

    Each set's first element is the rep; the rest are the non-rep
    members. Each member is `{"i": int, "snippet": str}`. The snippet
    is validated against `propositions[i]` via `_snippet_matches`;
    members whose snippets do not validate are dropped silently
    (taken as a signal that the LLM lost track of the index).

    Filtering applied:
      - Out-of-range or non-integer indices dropped.
      - Non-string or unmatched snippets dropped.
      - Within a single set, duplicate indices dropped.
      - Sets of size < 2 (after filtering) dropped.
      - If the rep itself is dropped during snippet validation, the
        set is dropped entirely (no rep => no anchor for Phase 1/2).

    Returns:
      list of int-index lists (rep first) or empty list if the
        envelope is recoverable but no sets survive;
      None if the JSON envelope is unrecoverable.
    """
    payload = _safe_json_loads(text)
    if not isinstance(payload, dict):
        return None
    sets = payload.get("sets")
    if not isinstance(sets, list):
        return None

    n = len(propositions)
    out: List[List[int]] = []
    for s in sets:
        if not isinstance(s, list) or len(s) < 1:
            continue
        cell: List[int] = []
        seen_in_cell: set = set()
        rep_dropped = False
        for pos, entry in enumerate(s):
            if not isinstance(entry, dict):
                if pos == 0:
                    rep_dropped = True
                continue
            idx = entry.get("i")
            snippet = entry.get("snippet")
            if not isinstance(idx, int) or idx < 0 or idx >= n:
                if pos == 0:
                    rep_dropped = True
                continue
            if not isinstance(snippet, str) or not _snippet_matches(
                snippet, propositions[idx]
            ):
                if pos == 0:
                    rep_dropped = True
                continue
            if idx in seen_in_cell:
                continue
            seen_in_cell.add(idx)
            cell.append(idx)
        if rep_dropped:
            continue
        if len(cell) >= 2:
            out.append(cell)

    return out


def partition_propositions(
    propositions: List[str],
    llm: LLMClient,
    cache: Cache,
    shard_size: int,
    max_class_size: int,
    step_b_mode: str = "exhaustive",
) -> List[List[str]]:
    """Step A. Identify candidate similarity sets among `propositions`.

    The LLM is asked to emit only sets it is CONFIDENT contain
    near-paraphrases — high precision, low recall by design. Most
    propositions will not appear in any set, and that is correct;
    they enter Step C as singletons. Overlap across sets is allowed
    (a proposition may be a near-paraphrase of distinct claims).

    Two prompt modes, dispatched on `step_b_mode`:

      - "exhaustive" (default): bare-index output
        `{"sets": [[i, j, ...], ...]}`. Order within a set carries no
        meaning. Backward-compatible with all existing callers.

      - "pivot": snippet-grounded output
        `{"sets": [[{"i": int, "snippet": str}, ...], ...]}` with
        per-member snippet validation against the indexed
        proposition. The first element of each set is the LLM's
        designated rep. Returned class members preserve that
        ordering: the first string of each returned class is the
        rep, used by the pivot Step B as an anchor.

    For len(propositions) > shard_size, propositions are sharded by
    sorted index (deterministic). Cross-shard near-paraphrases cannot
    be discovered in this pass — accept the recall loss for very
    large inputs and re-run with a different shard ordering if needed.

    Side effects:
        - One cached LLM call per shard (plus one retry on parse
          failure).
    """
    if not propositions:
        return []

    sorted_props = sorted(propositions)
    shards: List[List[str]] = [
        sorted_props[i : i + shard_size]
        for i in range(0, len(sorted_props), shard_size)
    ]

    all_classes: List[List[str]] = []
    for shard_idx, shard in enumerate(shards):
        if step_b_mode == "pivot":
            prompt = _build_similarity_sets_prompt_pivot(shard, max_class_size)
        else:
            prompt = _build_similarity_sets_prompt(shard, max_class_size)
        response = cached_ask(llm, prompt, cache)
        if step_b_mode == "pivot":
            sets = _parse_similarity_sets_response_pivot(response, shard)
        else:
            sets = _parse_similarity_sets_response(response, len(shard))

        if sets is None:
            # One retry with the malformed output appended for context.
            response = cached_ask(llm, _retry_prompt(prompt, response), cache)
            if step_b_mode == "pivot":
                sets = _parse_similarity_sets_response_pivot(response, shard)
            else:
                sets = _parse_similarity_sets_response(response, len(shard))

        if sets is None:
            print(
                f"[quotient] Step A shard {shard_idx + 1}/{len(shards)} "
                "failed to parse; emitting no candidate sets for this shard.",
                flush=True,
            )
            sets = []

        print(
            f"[quotient] Step A shard {shard_idx + 1}/{len(shards)}: "
            f"{len(sets)} candidate set(s) returned "
            f"(sizes: {[len(s) for s in sets[:10]]}{'...' if len(sets) > 10 else ''}).",
            flush=True,
        )

        for cell_indices in sets:
            all_classes.append([shard[i] for i in cell_indices])

    if step_b_mode == "pivot":
        return _split_oversized_pivot(all_classes, max_class_size)
    return _split_oversized(all_classes, max_class_size)


# =====================================================================
# Step B — Per-class edge expansion
# =====================================================================
#
# Three prompts in CONFIG["prompts"]:
#   - step_b_exhaustive       (single call per class, full cell grid)
#   - step_b_phase1_lift      (pivot Phase 1: non-rep edges → rep)
#   - step_b_phase2_distribute (pivot Phase 2: rep edges → non-reps)


def _format_member_with_context(
    idx: int, member: str, context: Optional[Dict[str, Any]],
) -> str:
    """Render a single member with its upstream Module 3 context.
    `context` may be None for members that have no recorded provenance
    (e.g. propositions known only as expansion-edge endpoints)."""
    lines = [f"{idx}: {member}"]
    if context:
        topic = context.get("topic") or ""
        path = context.get("path") or []
        question = context.get("question") or ""
        statement = context.get("statement") or ""
        path_str = " -> ".join(path) if path else ""
        if topic:
            lines.append(f"   topic: {topic}")
        if path_str:
            lines.append(f"   path:  {path_str}")
        if question:
            lines.append(f"   question:  {question}")
        if statement:
            lines.append(f"   statement: {statement}")
    return "\n".join(lines)


def _build_expansion_prompt(
    members: List[str],
    member_contexts: List[Optional[Dict[str, Any]]],
    probes: List[str],
    relations: List[str],
) -> str:
    numbered_members_with_context = "\n".join(
        _format_member_with_context(i, m, ctx)
        for i, (m, ctx) in enumerate(zip(members, member_contexts))
    )
    numbered_probes = "\n".join(f"{i}: {z}" for i, z in enumerate(probes))
    return _format_prompt(
        "step_b_exhaustive",
        numbered_members_with_context=numbered_members_with_context,
        numbered_probes=numbered_probes,
        rel_list=", ".join(relations),
    )


def _parse_expansion_response(
    text: str,
    n_members: int,
    n_probes: int,
    relations: List[str],
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Validate cells. Returns (kept_cells, n_dropped) on a parseable
    response, or None on an unparseable response. The empty-but-valid
    case returns ([], 0); the caller distinguishes that from None and
    only retries on None.

    Each cell is the LLM asserting that for the specific (member,
    relation, probe, direction), the causal claim holds. Each member
    receives an independent judgment — symmetry across members is the
    LLM's job, not ours."""
    payload = _safe_json_loads(text)
    if not isinstance(payload, dict):
        return None
    asserted = payload.get("asserted")
    if not isinstance(asserted, list):
        return None

    relation_set = set(relations)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    seen: set = set()
    for cell in asserted:
        if not isinstance(cell, dict):
            dropped += 1
            continue
        m = cell.get("m")
        z = cell.get("z")
        r = cell.get("r")
        d = cell.get("d")
        if (
            not isinstance(m, int) or m < 0 or m >= n_members
            or not isinstance(z, int) or z < 0 or z >= n_probes
            or not isinstance(r, str) or r not in relation_set
            or d not in ("out", "in")
        ):
            dropped += 1
            continue
        # Dedup identical cells from a chatty LLM.
        key = (m, r, z, d)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"m": m, "z": z, "r": r, "d": d})
    return (kept, dropped)


def _node_degrees(edges: List[Edge]) -> Dict[str, int]:
    """Total in+out degree per normalized node string. Used to rank
    probes deterministically when truncation is required."""
    deg: Counter = Counter()
    for e in edges:
        deg[_norm(e["subj"])] += 1
        deg[_norm(e["obj"])] += 1
    return dict(deg)


def _node_primary_domain(edges: List[Edge]) -> Dict[str, str]:
    """Most common domain per normalized node, with alphabetical
    tiebreak. Used to assign a domain to expansion edges."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for e in edges:
        counts[_norm(e["subj"])][e["domain"]] += 1
        counts[_norm(e["obj"])][e["domain"]] += 1
    out: Dict[str, str] = {}
    for node, c in counts.items():
        # Sort first by frequency descending, then by domain string
        # ascending for determinism.
        ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        out[node] = ranked[0][0]
    return out


# ---------------------------------------------------------------------
# Pivot-mode Step B (two LLM calls per candidate class)
# ---------------------------------------------------------------------


def _member_edges_for_prompt(
    member: str, edges: List[Edge],
) -> List[Tuple[str, str, str]]:
    """Return a deterministically-sorted list of (direction, rel,
    neighbour_surface) tuples for `member` (matched on normalised
    form).

    Sorting key is (direction, rel, normalised-neighbour) for
    cache-stability across runs that may vary the input edge order.
    The displayed neighbour is the longest surface form observed for
    a given normalised key, matching the rep-selection convention
    used elsewhere in this module."""
    member_norm = _norm(member)
    by_key: Dict[Tuple[str, str, str], str] = {}
    for e in edges:
        s_norm = _norm(e["subj"])
        o_norm = _norm(e["obj"])
        if s_norm == member_norm and o_norm != member_norm:
            key = ("out", e["rel"], o_norm)
            surface = e["obj"]
            if key not in by_key or len(surface) > len(by_key[key]):
                by_key[key] = surface
        elif o_norm == member_norm and s_norm != member_norm:
            key = ("in", e["rel"], s_norm)
            surface = e["subj"]
            if key not in by_key or len(surface) > len(by_key[key]):
                by_key[key] = surface
    out: List[Tuple[str, str, str]] = []
    for key in sorted(by_key.keys()):
        direction, rel, _ = key
        out.append((direction, rel, by_key[key]))
    return out


def _format_edges_numbered(edges: List[Tuple[str, str, str]]) -> str:
    """One numbered line per edge, indented for embedding under a
    member block: `     0: (out, causes, "neighbour")`."""
    if not edges:
        return "     (none)"
    return "\n".join(
        f'     {i}: ({d}, {r}, "{n}")' for i, (d, r, n) in enumerate(edges)
    )


def _format_member_with_edges(
    idx: int,
    member: str,
    context: Optional[Dict[str, Any]],
    edges: List[Tuple[str, str, str]],
) -> str:
    """Render a member block (header + context lines) with its edges
    appended as a numbered list under an "edges:" sub-header."""
    head = _format_member_with_context(idx, member, context)
    return head + "\n   edges:\n" + _format_edges_numbered(edges)


def _parse_pivot_phase_response(
    text: str,
    n_members: int,
    max_e_for_member: List[int],
) -> Optional[Tuple[List[Dict[str, int]], int]]:
    """Parse a Phase 1 / Phase 2 response into kept cells.

    Each input cell: `{"m": int, "e": int, "holds": bool}`.
    `max_e_for_member[m]` gives the upper bound on `e` for that
    member.

    Cells with bad indices, non-boolean `holds`, or duplicates are
    dropped (counted toward `n_dropped`); cells with `holds: false`
    are dropped silently (not hallucinations, just no-asserts). Only
    `holds: true` cells flow into `kept`.

    Returns:
      (kept_cells, n_dropped) on a parseable response;
      None on an unparseable envelope (caller should retry once).
    """
    payload = _safe_json_loads(text)
    if not isinstance(payload, dict):
        return None
    asserted = payload.get("asserted")
    if not isinstance(asserted, list):
        return None

    kept: List[Dict[str, int]] = []
    dropped = 0
    seen: set = set()
    for cell in asserted:
        if not isinstance(cell, dict):
            dropped += 1
            continue
        m = cell.get("m")
        e = cell.get("e")
        holds = cell.get("holds")
        if (
            not isinstance(m, int) or m < 0 or m >= n_members
            or not isinstance(e, int)
            or e < 0
            or m >= len(max_e_for_member)
            or e >= max_e_for_member[m]
            or not isinstance(holds, bool)
        ):
            dropped += 1
            continue
        key = (m, e)
        if key in seen:
            continue
        seen.add(key)
        if holds:
            kept.append({"m": m, "e": e})
    return (kept, dropped)


def _phase1_lift(
    rep: str,
    rep_context: Optional[Dict[str, Any]],
    others: List[str],
    others_contexts: List[Optional[Dict[str, Any]]],
    others_edges: List[List[Tuple[str, str, str]]],
    relations: List[str],
    llm: LLMClient,
    cache: Cache,
) -> Tuple[List[Tuple[str, str, str]], int]:
    """Phase 1: ask the LLM, for each (non-rep member, member-edge)
    cell, whether the analogous edge holds for the rep.

    Returns:
      (lifted_edges, n_dropped) where lifted_edges is a sorted,
      deduplicated list of (direction, rel, neighbour) tuples to add
      to the rep's edge set, and n_dropped is the hallucination count
      from response parsing.

    No LLM call is issued (and ([], 0) is returned) when there are no
    non-rep members or none of the non-rep members has any edges.
    """
    if not others:
        return ([], 0)
    if all(not es for es in others_edges):
        return ([], 0)

    numbered = "\n".join(
        _format_member_with_edges(i, m, ctx, es)
        for i, (m, ctx, es) in enumerate(
            zip(others, others_contexts, others_edges)
        )
    )
    prompt = _format_prompt(
        "step_b_phase1_lift",
        rep_block=_format_member_with_context(0, rep, rep_context),
        numbered_members_with_edges=numbered,
        rel_list=", ".join(relations),
    )
    response = cached_ask(llm, prompt, cache)
    max_e = [len(es) for es in others_edges]
    parsed = _parse_pivot_phase_response(response, len(others), max_e)
    if parsed is None:
        response = cached_ask(llm, _retry_prompt(prompt, response), cache)
        parsed = _parse_pivot_phase_response(response, len(others), max_e)
    if parsed is None:
        return ([], 0)
    cells, dropped = parsed

    lifted_set: set = set()
    for cell in cells:
        m, e = cell["m"], cell["e"]
        lifted_set.add(others_edges[m][e])
    return (sorted(lifted_set), dropped)


def _phase2_distribute(
    rep: str,
    rep_context: Optional[Dict[str, Any]],
    rep_edges: List[Tuple[str, str, str]],
    others: List[str],
    others_contexts: List[Optional[Dict[str, Any]]],
    relations: List[str],
    llm: LLMClient,
    cache: Cache,
) -> Tuple[List[Tuple[int, Tuple[str, str, str]]], int]:
    """Phase 2: ask the LLM, in one combined prompt, which of the rep's
    enriched edges hold for each non-rep member.

    Returns:
      (distributed_pairs, n_dropped) where distributed_pairs is a list
      of (member_idx, edge_tuple) to add as expansion edges anchored on
      the corresponding member, and n_dropped is the hallucination
      count.

    No LLM call is issued when there are no non-rep members or the rep
    has no edges (after Phase 1 lift).
    """
    if not others or not rep_edges:
        return ([], 0)

    rep_block = (
        _format_member_with_context(0, rep, rep_context)
        + "\n   edges:\n"
        + _format_edges_numbered(rep_edges)
    )
    numbered = "\n".join(
        _format_member_with_context(i, m, ctx)
        for i, (m, ctx) in enumerate(zip(others, others_contexts))
    )
    prompt = _format_prompt(
        "step_b_phase2_distribute",
        rep_block_with_edges=rep_block,
        numbered_members_with_context=numbered,
        rel_list=", ".join(relations),
    )
    response = cached_ask(llm, prompt, cache)
    max_e = [len(rep_edges)] * len(others)
    parsed = _parse_pivot_phase_response(response, len(others), max_e)
    if parsed is None:
        response = cached_ask(llm, _retry_prompt(prompt, response), cache)
        parsed = _parse_pivot_phase_response(response, len(others), max_e)
    if parsed is None:
        return ([], 0)
    cells, dropped = parsed

    distributed: List[Tuple[int, Tuple[str, str, str]]] = []
    for cell in cells:
        m, e = cell["m"], cell["e"]
        distributed.append((m, rep_edges[e]))
    return (distributed, dropped)


def _expand_pivot(
    members: List[str],
    member_contexts: List[Optional[Dict[str, Any]]],
    edges: List[Edge],
    relations: List[str],
    llm: LLMClient,
    cache: Cache,
    skip_already_connected_pairs: bool = True,
) -> Tuple[List[Edge], int]:
    """Step B in pivot mode (two LLM calls per class).

    `members[0]` is the rep designated by Step A; `members[1:]` are
    the non-rep members. Phase 1 lifts non-rep members' extraction
    edges onto the rep (when the LLM agrees); Phase 2 distributes the
    rep's enriched edge set onto each non-rep member (when the LLM
    agrees).

    Returns (new_expansion_edges, n_hallucinations_dropped). The same
    `extracted` and `already-connected` filters apply as in
    exhaustive mode.
    """
    if len(members) < 2:
        return ([], 0)

    rep = members[0]
    rep_context = member_contexts[0] if member_contexts else None
    others = list(members[1:])
    others_contexts = list(member_contexts[1:]) if member_contexts else [None] * len(others)

    others_edges = [_member_edges_for_prompt(m, edges) for m in others]
    rep_extraction_edges = _member_edges_for_prompt(rep, edges)

    # Phase 1 — lift non-rep members' edges onto the rep.
    lifted, dropped_p1 = _phase1_lift(
        rep, rep_context, others, others_contexts, others_edges,
        relations, llm, cache,
    )

    # Rep's enriched edge list = extraction ∪ Phase-1 lifted.
    rep_enriched = sorted(set(rep_extraction_edges) | set(lifted))

    # Phase 2 — distribute rep's enriched edges onto non-rep members.
    distributed, dropped_p2 = _phase2_distribute(
        rep, rep_context, rep_enriched,
        others, others_contexts, relations, llm, cache,
    )

    # Materialise expansion Edge records, applying the same filters as
    # exhaustive mode.
    extracted_keys = {
        (_norm(e["subj"]), _norm(e["rel"]), _norm(e["obj"]))
        for e in edges
    }
    connected_pairs: set = set()
    if skip_already_connected_pairs:
        for e in edges:
            connected_pairs.add(frozenset({_norm(e["subj"]), _norm(e["obj"])}))
    primary_domain = _node_primary_domain(edges)

    new_edges: List[Edge] = []

    def _emit(anchor: str, edge_tuple: Tuple[str, str, str]) -> None:
        direction, rel, neighbour = edge_tuple
        if direction == "out":
            subj, obj = anchor, neighbour
        else:
            subj, obj = neighbour, anchor
        key = (_norm(subj), _norm(rel), _norm(obj))
        if key in extracted_keys:
            return
        if skip_already_connected_pairs:
            pair = frozenset({_norm(subj), _norm(obj)})
            if pair in connected_pairs:
                return
            connected_pairs.add(pair)
        domain = (
            primary_domain.get(_norm(subj))
            or primary_domain.get(_norm(obj))
            or "expansion"
        )
        new_edges.append(Edge(
            subj=subj,
            rel=rel,
            obj=obj,
            domain=domain,
            edge_source="expansion",
            provenance=[],
        ))

    for edge_tuple in lifted:
        _emit(rep, edge_tuple)
    for m, edge_tuple in distributed:
        _emit(others[m], edge_tuple)

    return (new_edges, dropped_p1 + dropped_p2)


# ---------------------------------------------------------------------
# Step B dispatcher and exhaustive implementation
# ---------------------------------------------------------------------


def expand_edges_for_class(
    members: List[str],
    member_contexts: List[Optional[Dict[str, Any]]],
    edges: List[Edge],
    relations: List[str],
    max_probes: int,
    llm: LLMClient,
    cache: Cache,
    skip_already_connected_pairs: bool = True,
    mode: str = "exhaustive",
) -> Tuple[List[Edge], int]:
    """Step B. Dispatches on `mode` to either exhaustive (one LLM
    call per class, full cell grid) or pivot (two LLM calls per
    class, rep-anchored expansion).

    Returns (new_expansion_edges, n_hallucinations_dropped).

    Under `mode="exhaustive"` (default), `max_probes` truncates the
    probe set; see `_expand_exhaustive`. Under `mode="pivot"`,
    `members[0]` is the rep designated by Step A and the probe set
    is determined by member edge lists; `max_probes` is unused. See
    `_expand_pivot`.
    """
    if mode == "pivot":
        return _expand_pivot(
            members=members,
            member_contexts=member_contexts,
            edges=edges,
            relations=relations,
            llm=llm,
            cache=cache,
            skip_already_connected_pairs=skip_already_connected_pairs,
        )
    if mode != "exhaustive":
        raise ValueError(f"unknown step_b_mode: {mode!r}")
    return _expand_exhaustive(
        members=members,
        member_contexts=member_contexts,
        edges=edges,
        relations=relations,
        max_probes=max_probes,
        llm=llm,
        cache=cache,
        skip_already_connected_pairs=skip_already_connected_pairs,
    )


def _expand_exhaustive(
    members: List[str],
    member_contexts: List[Optional[Dict[str, Any]]],
    edges: List[Edge],
    relations: List[str],
    max_probes: int,
    llm: LLMClient,
    cache: Cache,
    skip_already_connected_pairs: bool = True,
) -> Tuple[List[Edge], int]:
    """Step B. ONE LLM call per candidate class (exhaustive cell-grid
    enumeration; the original implementation).

    Returns (new_expansion_edges, n_hallucinations_dropped).

    Builds the union of 1-hop neighbourhoods of all `members`,
    truncates to `max_probes` (highest degree first; alphabetical
    tiebreak), asks the LLM to enumerate every (member, relation,
    probe, direction) for which the causal claim is true. Each member
    is presented with its upstream Module 3 context (topic, path,
    question, statement) so the LLM can disambiguate intended scope.
    New edges are returned with edge_source='expansion'. Edges that
    already exist as extraction edges are filtered out (no
    double-counting).

    When `skip_already_connected_pairs` is True, an LLM-asserted cell
    is also dropped if its (subj, obj) endpoints are ALREADY connected
    (in either direction, by any verb) by an extraction edge or by an
    expansion edge added earlier. This prevents the LLM from piling
    contradictory verbs onto a pair the document has already settled.

    `member_contexts` aligns with `members`: each entry is the
    Module 3 provenance dict for that member (or None if not
    available).

    Side effects:
        - One cached LLM call.
    """
    if len(members) < 2:
        # Singleton classes carry no comparative information; skip the
        # LLM call. They cannot trigger a merge anyway.
        return ([], 0)

    member_set = {_norm(m) for m in members}

    # 1-hop neighbourhood: every node connected to at least one member.
    neighbour_set: set = set()
    for e in edges:
        s, o = _norm(e["subj"]), _norm(e["obj"])
        if s in member_set:
            neighbour_set.add(o)
        if o in member_set:
            neighbour_set.add(s)
    neighbour_set -= member_set  # do not probe a member against itself

    if not neighbour_set:
        return ([], 0)

    # Truncate to max_probes by degree.
    deg = _node_degrees(edges)
    probes_sorted = sorted(
        neighbour_set,
        key=lambda n: (-deg.get(n, 0), n),
    )
    probes = probes_sorted[:max_probes]

    prompt = _build_expansion_prompt(members, member_contexts, probes, relations)
    response = cached_ask(llm, prompt, cache)
    parsed = _parse_expansion_response(
        response, len(members), len(probes), relations,
    )

    if parsed is None:
        # Unparseable. Try once more with the malformed output as context.
        response = cached_ask(llm, _retry_prompt(prompt, response), cache)
        parsed = _parse_expansion_response(
            response, len(members), len(probes), relations,
        )

    if parsed is None:
        # Both attempts unparseable; drop the call (no expansion edges,
        # no hallucinations counted).
        cells, dropped = [], 0
    else:
        cells, dropped = parsed

    # Build edge set for fast existence checks.
    extracted_keys = {
        (_norm(e["subj"]), _norm(e["rel"]), _norm(e["obj"]))
        for e in edges
    }
    # Undirected node-pair set for the "already connected" check.
    # Includes both extraction and prior expansion edges so the
    # check accumulates across Step B calls within a run.
    connected_pairs: set = set()
    if skip_already_connected_pairs:
        for e in edges:
            connected_pairs.add(frozenset({_norm(e["subj"]), _norm(e["obj"])}))

    primary_domain = _node_primary_domain(edges)
    new_edges: List[Edge] = []
    for cell in cells:
        m_str = members[cell["m"]]
        z_str = probes[cell["z"]]
        r_str = cell["r"]
        if cell["d"] == "out":
            subj, obj = m_str, z_str
        else:
            subj, obj = z_str, m_str

        # Skip if this exact triple already exists.
        key = (_norm(subj), _norm(r_str), _norm(obj))
        if key in extracted_keys:
            continue

        # Skip if the node pair is already connected (any direction,
        # any verb). Prevents pile-on of alternative verbs onto pairs
        # already covered by extraction or earlier expansion.
        if skip_already_connected_pairs:
            pair = frozenset({_norm(subj), _norm(obj)})
            if pair in connected_pairs:
                continue
            connected_pairs.add(pair)

        # Pick the domain from the subject side; fall back to the
        # object side; finally to "expansion" if neither has one.
        domain = (
            primary_domain.get(_norm(subj))
            or primary_domain.get(_norm(obj))
            or "expansion"
        )
        new_edges.append(Edge(
            subj=subj,
            rel=r_str,
            obj=obj,
            domain=domain,
            edge_source="expansion",
            provenance=[],
        ))

    return (new_edges, dropped)


# =====================================================================
# Step C — Structural quotient
# =====================================================================

def _canonical_rel(
    r: str, equivalence_map: Optional[Dict[str, str]],
) -> str:
    """Map a raw relation verb to its equivalence-class label.
    Verbs not in the map are their own class — this is reflexive and
    safe when the map is partial or empty."""
    if not equivalence_map:
        return r
    return equivalence_map.get(r, r)


def _property_set(
    node: str,
    out_index: Dict[str, set],
    in_index: Dict[str, set],
) -> frozenset:
    """prop(P) = {('out', r, z) : (P,r,z) ∈ E} ∪ {('in', r, z) : (z,r,P) ∈ E}.

    The relation labels in this set are CANONICAL (post-equivalence)
    — see structural_equivalence for the mapping. Returns a frozenset
    suitable for hashing into a class key."""
    out_facts = {("out", r, z) for r, z in out_index.get(node, set())}
    in_facts = {("in", r, z) for r, z in in_index.get(node, set())}
    return frozenset(out_facts | in_facts)


def structural_equivalence(
    propositions: List[str],
    edges: List[Edge],
    rng: np.random.Generator,
    relation_equivalence: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Step C. Compute ~ defined by equality of property sets and
    return a map P → representative of [P]_~.

    `relation_equivalence` (verb -> class label) coarsens the relation
    set before computing prop(P). Two edges whose verbs share a class
    contribute the same property and so collapse for identity purposes.
    A None or empty map preserves raw verbs as their own classes (the
    no-softening behavior).

    The representative is the longest member string. Ties are broken
    deterministically by passing the candidates through `rng` once;
    rng is the only source of randomness in the module and is seeded
    from CONFIG['seed'].

    Pure computation; no LLM, no I/O.
    """
    # Build directed neighbour indices keyed by normalized strings.
    # Relations are mapped to their equivalence class first, so the
    # property-set check operates on canonical labels.
    out_index: Dict[str, set] = defaultdict(set)  # subj_norm -> {(rel_canon, obj_norm)}
    in_index: Dict[str, set] = defaultdict(set)   # obj_norm  -> {(rel_canon, subj_norm)}
    for e in edges:
        s = _norm(e["subj"])
        o = _norm(e["obj"])
        r = _canonical_rel(_norm(e["rel"]), relation_equivalence)
        out_index[s].add((r, o))
        in_index[o].add((r, s))

    # Group propositions by property-set key. Operate on the original
    # (un-normalized) strings as members so we can preserve the
    # surface phrasing in the output schema; key by normalized form.
    norm_to_members: Dict[str, List[str]] = defaultdict(list)
    for p in propositions:
        norm_to_members[_norm(p)].append(p)

    classes: Dict[frozenset, List[str]] = defaultdict(list)
    for p_norm, members in norm_to_members.items():
        key = _property_set(p_norm, out_index, in_index)
        classes[key].extend(members)

    # Pick representative per class. Longest by length, ties broken
    # via rng (deterministic given the seed).
    rep_of: Dict[str, str] = {}
    for members in classes.values():
        max_len = max(len(m) for m in members)
        candidates = sorted(m for m in members if len(m) == max_len)
        if len(candidates) == 1:
            rep = candidates[0]
        else:
            rep = candidates[int(rng.integers(0, len(candidates)))]
        for m in members:
            rep_of[m] = rep

    return rep_of


# =====================================================================
# Output construction
# =====================================================================

def _assign_group_ids(rep_of: Dict[str, str]) -> Dict[str, str]:
    """Assign deterministic group IDs (g_000, g_001, ...) keyed on
    the representative string. Sorted alphabetically for stability."""
    reps_sorted = sorted(set(rep_of.values()))
    width = max(3, len(str(len(reps_sorted))))
    return {
        rep: f"g_{i:0{width}d}"
        for i, rep in enumerate(reps_sorted)
    }


def _members_by_rep(rep_of: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for member, rep in rep_of.items():
        out[rep].append(member)
    for rep in out:
        out[rep] = sorted(out[rep])
    return out


def _build_output_records(
    extracted: List[Edge],
    expansion: List[Edge],
    rep_of: Dict[str, str],
    provenance_truncate: int,
    relation_equivalence: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate raw edges into one record per quotient edge.

    Quotient-edge identity: (rep(subj_norm), canonical_rel,
    rep(obj_norm)) — i.e. edges with raw verbs in the same equivalence
    class collapse into a single record. The output `rel` field carries
    the canonical class label; `original_rels` preserves the raw verbs
    that contributed (sorted, deduplicated).
    """
    rep_of_norm: Dict[str, str] = {_norm(k): v for k, v in rep_of.items()}
    group_ids = _assign_group_ids(rep_of)
    members = _members_by_rep(rep_of)

    # Aggregate by quotient-edge key (using canonical relation).
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for e in (*extracted, *expansion):
        s_norm = _norm(e["subj"])
        o_norm = _norm(e["obj"])
        r_norm = _norm(e["rel"])
        r_canon = _canonical_rel(r_norm, relation_equivalence)
        rep_s = rep_of_norm.get(s_norm)
        rep_o = rep_of_norm.get(o_norm)
        if rep_s is None or rep_o is None:
            # Should not happen if proposition set was built consistently.
            continue
        key = (rep_s, r_canon, rep_o)
        bucket = grouped.setdefault(key, {
            "subj": rep_s,
            "rel": r_canon,
            "obj": rep_o,
            "domain_counter": Counter(),
            "sources": set(),
            "original_rels": set(),
            "provenance_all": [],
        })
        bucket["domain_counter"][e["domain"]] += 1
        bucket["sources"].add(e["edge_source"])
        bucket["original_rels"].add(r_norm)
        bucket["provenance_all"].extend(e["provenance"])

    out_records: List[Dict[str, Any]] = []
    for (rep_s, r_canon, rep_o), bucket in grouped.items():
        domain = bucket["domain_counter"].most_common(1)[0][0]
        prov_all = bucket["provenance_all"]
        prov_keep = prov_all[:provenance_truncate]
        record = {
            "subj": rep_s,
            "rel": r_canon,
            "obj": rep_o,
            "domain": domain,
            "edge_source":   sorted(bucket["sources"]),
            "original_rels": sorted(bucket["original_rels"]),
            "group_id_subj": group_ids[rep_s],
            "group_id_obj":  group_ids[rep_o],
            "members_subj":  members[rep_s],
            "members_obj":   members[rep_o],
            "provenance":    prov_keep,
            "provenance_truncated": len(prov_all) > len(prov_keep),
            "provenance_total_count": len(prov_all),
        }
        out_records.append(record)

    # Stable order for byte-comparison across runs.
    out_records.sort(key=lambda r: (r["subj"], r["rel"], r["obj"]))
    return out_records


# =====================================================================
# End-to-end orchestration
# =====================================================================

def _triples_to_edges(triples: List[Dict[str, Any]]) -> List[Edge]:
    """Convert input records into Edge dicts.

    Accepts both Module-4 raw triples (top-level `topic`, `path`,
    `question`, `statement`; `rel` carries the raw causal verb) AND
    quotient-output records from a prior iteration (with `original_rels`
    listing the raw verbs that contributed and `rel` carrying the
    canonical class label applied by Step C).

    For Module-4 raw triples: one Edge per input row, `rel` = raw verb,
    `edge_source` = "extraction" (raw extraction is the only possible
    source at iteration 0).

    For quotient-output records: ONE EDGE PER (raw verb × source label).
    Two reasons for the cross product:

    1. Per-verb expansion. `t["rel"]` carries the canonical class label
       (post-softening), but the raw verbs were preserved in
       `original_rels`. Iterating over `original_rels` keeps the real
       verbs flowing through Step B and Step C, so
       `relation_equivalence` stays a Step-C-only coarsening rather
       than permanently erasing verb-level semantics from iteration 2
       onward.

    2. Per-source expansion. A prior-iteration bucket's `edge_source`
       is a list (e.g., ``["extraction", "expansion"]`` for an edge
       that received contributions from both raw extraction and an
       LLM-asserted expansion). We emit one Edge per source label so
       the next iteration's bucketing — which calls
       ``bucket["sources"].add(e["edge_source"])`` on a single string
       — can re-union the labels into the new bucket. Without this,
       the previous list would be silently overwritten to a single
       value before bucketing, destroying the lineage and inflating
       the "extraction-only" count under iteration. (See the
       "Iteration N inflates extraction count" issue: at N=5 in v8,
       this caused 1696 of 3074 edges to be misclassified as
       extraction-only when most had pure-expansion lineage in
       earlier passes.)
    """
    out: List[Edge] = []
    for t in triples:
        if isinstance(t.get("provenance"), list) and t["provenance"]:
            provenance = list(t["provenance"])
        else:
            provenance = [{
                "topic":     t.get("topic", ""),
                "path":      t.get("path", []),
                "question":  t.get("question", ""),
                "statement": t.get("statement", ""),
            }]
        original_rels = t.get("original_rels")
        if isinstance(original_rels, list) and original_rels:
            verbs = list(original_rels)
        else:
            verbs = [t["rel"]]
        domain = t.get("domain", t.get("topic", ""))
        prior_source = t.get("edge_source")
        if isinstance(prior_source, list) and prior_source:
            sources = list(prior_source)
        elif isinstance(prior_source, str):
            sources = [prior_source]
        else:
            sources = ["extraction"]
        for raw_rel in verbs:
            for src in sources:
                out.append(Edge(
                    subj=t["subj"],
                    rel=raw_rel,
                    obj=t["obj"],
                    domain=domain,
                    edge_source=src,
                    provenance=provenance,
                ))
    return out


def _filter_uncorroborated_expansion_edges(
    expansion_per_class: List[Tuple[List[str], List[Edge]]],
    extracted: List[Edge],
    relation_equivalence: Optional[Dict[str, str]] = None,
) -> Tuple[List[Edge], int]:
    """Drop LLM-asserted expansion edges that no other member of the
    same candidate set agrees with.

    Cells are keyed by (canonical_relation, neighbour, direction) —
    the same key Step C uses to compute prop(P) — so verbs that share
    a relation-equivalence class corroborate one another. An edge is
    kept iff its cell has ≥ 2 distinct members of SOME candidate set
    contributing (counting both extraction and LLM-asserted edges).

    Extraction edges are not subject to filtering; only expansion
    edges are.

    Returns (kept_expansion_edges, n_uncorroborated_dropped). The
    drop count is computed against deduplicated keys, so two
    expansion edges with the same (subj, rel, obj) — possible when a
    proposition appears in multiple candidate sets — count as one
    edge for both kept and dropped totals.
    """
    # Pass 1: collect cells corroborated in any candidate class.
    corroborated_cells: set = set()
    for members, new_edges in expansion_per_class:
        member_set = {_norm(m) for m in members}
        cell_members: Dict[Tuple[str, str, str], set] = defaultdict(set)
        for e in (*extracted, *new_edges):
            s, o = _norm(e["subj"]), _norm(e["obj"])
            r_canon = _canonical_rel(_norm(e["rel"]), relation_equivalence)
            if s in member_set:
                cell_members[(r_canon, o, "out")].add(s)
            if o in member_set:
                cell_members[(r_canon, s, "in")].add(o)
        for cell, ms in cell_members.items():
            if len(ms) >= 2:
                corroborated_cells.add(cell)

    # Pass 2: keep expansion edges whose cell is corroborated; dedupe
    # across overlapping classes by (subj_norm, rel_norm, obj_norm).
    kept: List[Edge] = []
    seen_keys: set = set()
    all_input_keys: set = set()
    for members, new_edges in expansion_per_class:
        member_set = {_norm(m) for m in members}
        for e in new_edges:
            s, o = _norm(e["subj"]), _norm(e["obj"])
            raw_key = (s, _norm(e["rel"]), o)
            all_input_keys.add(raw_key)

            if raw_key in seen_keys:
                continue

            r_canon = _canonical_rel(_norm(e["rel"]), relation_equivalence)
            if s in member_set:
                cell = (r_canon, o, "out")
            elif o in member_set:
                cell = (r_canon, s, "in")
            else:
                # Defensive: not anchored on a member of this class —
                # keep, since we cannot judge corroboration here.
                seen_keys.add(raw_key)
                kept.append(e)
                continue

            if cell in corroborated_cells:
                seen_keys.add(raw_key)
                kept.append(e)

    n_dropped = len(all_input_keys) - len(seen_keys)
    return (kept, n_dropped)


def _compose_member_history(
    records: List[Dict[str, Any]],
    prev_history: Dict[str, set],
) -> Dict[str, set]:
    """Update the per-representative history map after one quotient
    iteration.

    `prev_history` maps each surface string to the set of *original*
    (iteration-0) propositions that flow into it. The returned map
    composes the current iteration's `members_subj` / `members_obj`
    lists with `prev_history`, so each new representative knows the
    full set of original propositions it absorbs through the chain
    of merges.

    Used only when config['max_iterations'] > 1."""
    new_history: Dict[str, set] = {}
    for rec in records:
        for side in ("subj", "obj"):
            rep = rec.get(side, "") or ""
            if not rep:
                continue
            members = rec.get(f"members_{side}") or [rep]
            originals: set = set()
            for m in members:
                if m in prev_history:
                    originals.update(prev_history[m])
                else:
                    originals.add(m)
            if rep in new_history:
                new_history[rep].update(originals)
            else:
                new_history[rep] = originals
    return new_history


def _build_node_provenance(
    triples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build a node -> upstream-context map from raw Module 4 records.

    Each proposition (normalized string) gets the FIRST extraction
    record that mentions it (as either subj or obj). The chosen
    record's topic / path / question / statement become the upstream
    context shown to the LLM in Step B's prompt."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in triples:
        ctx = {
            "topic":     t.get("topic", ""),
            "path":      t.get("path", []),
            "question":  t.get("question", ""),
            "statement": t.get("statement", ""),
        }
        for surface in (t.get("subj", ""), t.get("obj", "")):
            key = _norm(surface)
            if key and key not in out:
                out[key] = ctx
    return out


def quotient_triples(
    triples: List[Dict[str, Any]],
    llm: LLMClient,
    cache: Cache,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """End-to-end. Composes Steps A → B → C and emits the quotiented
    triples list together with a small stats dict.

    Returns (output_records, stats) where stats has keys:
        n_propositions, n_extracted, n_expansion, n_classes,
        n_hallucinations_dropped.

    Side effects:
        - LLM network calls (one per partition shard plus one per
          candidate class).
        - Cache reads/writes via `cache`.
    """
    extracted = _triples_to_edges(triples)
    node_provenance = _build_node_provenance(triples)

    # Proposition set: union of all subj/obj surface strings, deduplicated
    # by normalized form. We keep the longest representative of each
    # normalized cluster so prompt text remains informative.
    by_norm: Dict[str, str] = {}
    for e in extracted:
        for surface in (e["subj"], e["obj"]):
            n = _norm(surface)
            if n not in by_norm or len(surface) > len(by_norm[n]):
                by_norm[n] = surface
    propositions = sorted(by_norm.values())

    # Step A
    step_b_mode = config.get("step_b_mode", "exhaustive")
    classes = partition_propositions(
        propositions=propositions,
        llm=llm,
        cache=cache,
        shard_size=config["shard_size"],
        max_class_size=config["max_class_size"],
        step_b_mode=step_b_mode,
    )

    # Step B — track expansion per class so the optional corroboration
    # filter can group cell evidence by candidate set.
    expansion_per_class: List[Tuple[List[str], List[Edge]]] = []
    n_hallucinations = 0
    for cls_idx, members in enumerate(classes):
        member_contexts = [node_provenance.get(_norm(m)) for m in members]
        edges_so_far = extracted + [
            e for _m, es in expansion_per_class for e in es
        ]
        new_edges, dropped = expand_edges_for_class(
            members=members,
            member_contexts=member_contexts,
            edges=edges_so_far,
            relations=config["relations"],
            max_probes=config["max_probes_per_class"],
            llm=llm,
            cache=cache,
            skip_already_connected_pairs=config.get(
                "skip_already_connected_pairs", True
            ),
            mode=step_b_mode,
        )
        expansion_per_class.append((members, new_edges))
        n_hallucinations += dropped
        print(
            f"[quotient] Step B class {cls_idx + 1}/{len(classes)}: "
            f"|members|={len(members)}, +{len(new_edges)} expansion edges, "
            f"{dropped} dropped.",
            flush=True,
        )

    relation_equivalence = config.get("relation_equivalence")

    # Optional corroboration filter: drop solo LLM assertions before
    # they enter Step C and the output. Behavior unchanged when
    # config['corroboration_filter'] is False.
    if config.get("corroboration_filter", False):
        expansion, n_uncorroborated = _filter_uncorroborated_expansion_edges(
            expansion_per_class, extracted, relation_equivalence,
        )
        raw_total = len(set(
            (_norm(e["subj"]), _norm(e["rel"]), _norm(e["obj"]))
            for _m, es in expansion_per_class for e in es
        ))
        print(
            f"[quotient] Corroboration filter: kept {len(expansion)} of "
            f"{raw_total} unique expansion edges; "
            f"{n_uncorroborated} uncorroborated dropped.",
            flush=True,
        )
    else:
        expansion = [e for _m, es in expansion_per_class for e in es]
        n_uncorroborated = 0

    # Step C — relation equivalence is read from config (verb -> class
    # label). Edges with raw verbs in the same class collapse for the
    # Leibnizian identity check.
    rep_of = structural_equivalence(
        propositions=list(by_norm.values()),
        edges=extracted + expansion,
        rng=rng,
        relation_equivalence=relation_equivalence,
    )

    # Output construction (uses the same equivalence so that the
    # output's quotient-edge identity matches the structural check).
    records = _build_output_records(
        extracted=extracted,
        expansion=expansion,
        rep_of=rep_of,
        provenance_truncate=config["provenance_truncate"],
        relation_equivalence=relation_equivalence,
    )

    stats = {
        "n_propositions": len(by_norm),
        "n_extracted":    len(extracted),
        "n_expansion":    len(expansion),
        "n_uncorroborated_dropped": n_uncorroborated,
        "n_classes":      len(set(rep_of.values())),
        "n_hallucinations_dropped": n_hallucinations,
    }
    return (records, stats)


# =====================================================================
# CLI entry point
# =====================================================================

def main() -> None:
    """Read CONFIG, read input_filename, run quotient_triples, write
    output_filename. Reports a one-line summary on completion.

    Side effects:
        - Reads {input_filename} from cwd.
        - Writes {output_filename} to cwd.
        - Reads/writes the LLM disk cache under {cache_dir}.
    """
    config = CONFIG
    input_path = Path(config["input_filename"])
    output_path = Path(config["output_filename"])

    if not input_path.is_file():
        print(
            f"[quotient] FAIL: input not found at {input_path.resolve()}",
            file=sys.stderr,
        )
        sys.exit(1)

    triples: List[Dict[str, Any]] = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        triples.append(json.loads(raw))
    print(f"[quotient] Loaded {len(triples)} triples from {input_path}.")

    llm = make_llm_client(
        max_tokens=config["max_tokens_partition"],
        max_batch_size=config["batch_size"],
    )

    # Resolve model and temperature for cache binding. Read the same
    # env vars the LLM client reads, with the same defaults; if env
    # vars are absent, the LLM client's internal defaults apply, and
    # the cache key encodes whatever values we resolve here. Mismatches
    # cause cache misses, never silent staleness.
    import os
    model = os.environ.get("DEMOC_LLM_MODEL", "gpt-4.1-mini")
    temperature = float(os.environ.get("DEMOC_LLM_TEMPERATURE", "0.7"))

    cache_dir: Optional[Path] = (
        Path(config["cache_dir"]) if config["use_cache"] else None
    )
    cache = Cache(dir=cache_dir, model=model, temperature=temperature)

    rng = np.random.default_rng(config["seed"])

    max_iterations = int(config.get("max_iterations", 1))
    if max_iterations < 1:
        print(
            f"[quotient] FAIL: max_iterations must be >= 1, got {max_iterations}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initial member history: identity. Each original proposition
    # surface maps to {itself}. Updated after each iteration so the
    # final output's members_subj/members_obj name iteration-0
    # propositions, not intermediate-iteration representatives.
    member_history: Dict[str, set] = {}
    for t in triples:
        for surface in (t.get("subj", ""), t.get("obj", "")):
            if surface:
                member_history.setdefault(surface, {surface})

    current_triples = triples
    records: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}
    prev_n_classes: Optional[int] = None

    for it in range(max_iterations):
        print(
            f"[quotient] === Iteration {it + 1}/{max_iterations} "
            f"on {len(current_triples)} input triples ===",
            flush=True,
        )
        records, stats = quotient_triples(
            triples=current_triples,
            llm=llm,
            cache=cache,
            rng=rng,
            config=config,
        )
        # Compose the iteration's merges into the global history.
        if max_iterations > 1:
            member_history = _compose_member_history(records, member_history)
        # Fixed-point detection: structural quotient is monotone, so
        # n_classes is non-increasing across iterations. If it does
        # not decrease, no further merges are possible at this V.
        if prev_n_classes is not None and stats["n_classes"] >= prev_n_classes:
            print(
                f"[quotient] Fixed point reached at iteration {it + 1} "
                f"(n_classes = {stats['n_classes']}). Stopping early.",
                flush=True,
            )
            break
        prev_n_classes = stats["n_classes"]
        if it + 1 < max_iterations:
            current_triples = records

    # Rewrite final records' member lists from the composed history so
    # the panels and downstream consumers see iteration-0 propositions.
    if max_iterations > 1:
        for rec in records:
            for side in ("subj", "obj"):
                rep = rec.get(side, "") or ""
                if rep in member_history:
                    rec[f"members_{side}"] = sorted(member_history[rep])

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(json.dumps(record) + "\n")

    # ASCII-only output: Windows console default cp1252 chokes on the
    # Unicode arrow that reads more naturally in the design doc.
    print(
        f"[quotient] |V| {stats['n_propositions']} -> "
        f"|V/~| {stats['n_classes']}; "
        f"+{stats['n_expansion']} expansion edges (post-filter); "
        f"{stats['n_hallucinations_dropped']} dropped by hallucination guard; "
        f"{stats.get('n_uncorroborated_dropped', 0)} dropped as uncorroborated."
    )
    print(f"[quotient] Wrote {len(records)} grouped edges -> {output_path}")


if __name__ == "__main__":
    main()

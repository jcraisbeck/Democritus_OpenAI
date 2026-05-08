"""Centralized configuration for the indiscernibility quotient module.

All knobs for the three-step quotient pipeline (candidate partition,
per-class edge expansion, structural collapse) live here. The pipeline
orchestrator exposes only --quotient on the CLI; this file controls
everything else.

Read by:
  - scripts/indiscernibility_quotient.py
  - scripts/quotient_report.py
  - pipelines/pipeline_llm.py  (only the file-rename names)
"""

CONFIG = {
    # ---- File contracts (relative to the run directory) ----
    "input_filename":  "relational_triples.jsonl",
    "output_filename": "relational_triples_grouped.jsonl",
    "raw_backup_name": "relational_triples.raw.jsonl",

    # ---- Step A: candidate partition ----
    # Maximum propositions sent to the LLM in a single partition call.
    # Larger inputs are sharded; per-shard partitions are merged via
    # union-find on shared members.
    "shard_size":     500,
    # Hard ceiling on candidate-class size. Classes returned by the LLM
    # larger than this are deterministically split.
    "max_class_size": 30,

    # ---- Step B: per-class edge expansion ----
    # If the union of 1-hop neighbourhoods of a class exceeds this,
    # truncate to the highest-degree probes (deterministic ordering).
    # Only consulted under step_b_mode == "exhaustive".
    "max_probes_per_class": 60,

    # ---- Step B mode ----
    # "exhaustive": one LLM call per candidate class asking about every
    #               (member, relation, probe, direction) cell. Becomes
    #               unreliable on sets larger than ~5.
    # "pivot":      two LLM calls per candidate class. Phase 1 lifts
    #               edges from non-rep members onto the rep (the first
    #               element of the candidate set); Phase 2 distributes
    #               the rep's enriched edge set onto the non-rep
    #               members in one combined prompt. Step A's prompt
    #               additionally requires per-member snippet grounding
    #               and rep-first ordering when this mode is selected.
    "step_b_mode": "pivot",

    # ---- Output schema ----
    # Number of provenance records to retain per quotiented edge.
    # Total count is preserved in `provenance_total_count`.
    "provenance_truncate": 5,

    # ---- LLM disk cache ----
    "cache_dir": ".quotient_cache",
    "use_cache": True,

    # ---- Corroboration filter (Step C input) ----
    # When True, an LLM-asserted expansion edge contributes to prop(P)
    # only if at least one OTHER member of the same candidate set has
    # the same (canonical_relation, neighbour, direction) cell — either
    # via extraction or via another LLM assertion in this Step B call.
    # Solo LLM assertions (one-member-only) are dropped from the
    # property-set computation as LLM noise. Extraction edges are
    # always kept (document evidence).
    #
    # The motivation: a single LLM "yes" on a cell is one observation;
    # two independent "yes"es from members the LLM grouped as
    # near-paraphrases is much stronger evidence the cell is real.
    # Asymmetric per-cell answers are the dominant cause of failed
    # mergers in the strict Leibniz check.
    "corroboration_filter": False,

    # ---- Skip already-connected pairs in Step B ----
    # When True, expand_edges_for_class refuses to emit a new expansion
    # edge between two nodes that are already connected — by an
    # extraction edge or by an expansion edge added earlier in this
    # run. "Connected" is undirected and relation-agnostic: if any
    # edge in either direction with any verb already exists between
    # the two nodes, the new cell is dropped.
    #
    # Motivation: the LLM, when probed about every (member, relation,
    # probe, direction) cell, often asserts MULTIPLE relations between
    # the same node pair — sometimes contradictory ones (PROMOTES +
    # INHIBITS for the same edge). Letting them all through clutters
    # the visualization and dilutes the property sets used in Step C.
    # Extraction edges are document evidence; expansion should fill in
    # MISSING relations, not pile alternative verbs onto pairs the
    # document has already settled.
    "skip_already_connected_pairs": True,

    # ---- Reproducibility ----
    "seed": 0,

    # ---- Iteration ----
    # Number of times to compose the quotient with itself. Each
    # iteration re-runs Steps A -> B -> C on the previous iteration's
    # output, treating its representative strings as the new
    # proposition vocabulary. The structural quotient is monotone
    # (n_classes is non-increasing across iterations), so the loop
    # short-circuits when an iteration produces no further merges
    # (fixed point reached).
    #
    # 1 = single-pass (the original spec'd behavior).
    # >1 = iterative quotient, up to N passes.
    #
    # Members lists in the final output are *composed* across
    # iterations: each final representative's `members_subj` /
    # `members_obj` lists every iteration-0 proposition that maps to
    # it through the chain of merges. This keeps the "All grouped
    # propositions" panel showing original propositions, not
    # intermediate-iteration representatives.
    "max_iterations": 5,

    # ---- Closed vocabulary for the hallucination guard ----
    # Mirrors the keys of REL_PATTERNS in
    # scripts/relational_triple_extractor.py (built from
    # scripts/causal_verbs.rel_patterns()).
    "relations": [
        "causes",
        "leads_to",
        "increases",
        "reduces",
        "affects",
        "influences",
        "shapes",
        "contributes_to",
        "correlates_with",
        "is_associated_with",
    ],

    # ---- Relation equivalence (relation softening for Step C) ----
    # Maps each verb in `relations` to a canonical-class label. Step C
    # computes prop(P) using canonical labels rather than raw verbs, so
    # two edges whose relations share a class collapse for the
    # Leibnizian identity check. The output's `rel` field carries the
    # canonical class; the original verbs are preserved in
    # `original_rels` per output record.
    #
    # Default: Option 2 (sign-aware partition).
    #   PROMOTES   — positive directed influence
    #   INHIBITS   — negative directed influence
    #   ASSOCIATES — symmetric / non-causal
    #
    # To collapse all causal verbs into a single class (Option 1,
    # maximum compression, loses sign), map every key to one label.
    # To use raw verb identity (no softening), map each verb to itself.
    # "relation_equivalence": {
    #     "causes":             "PROMOTES",
    #     "leads_to":           "PROMOTES",
    #     "increases":          "PROMOTES",
    #     "contributes_to":     "PROMOTES",
    #     "affects":            "PROMOTES",
    #     "influences":         "PROMOTES",
    #     "shapes":             "PROMOTES",
    #     "reduces":            "INHIBITS",
    #     "correlates_with":    "ASSOCIATES",
    #     "is_associated_with": "ASSOCIATES",
    # },
        "relation_equivalence": { # Option 3: ASSOCIATION ONLY; PRESERVES CONNECTIONS AND DIRECTION BUT NOT QUALITY OF RELATIONSHIP
        "causes":             "ASSOCIATES",
        "leads_to":           "ASSOCIATES",
        "increases":          "ASSOCIATES",
        "contributes_to":     "ASSOCIATES",
        "affects":            "ASSOCIATES",
        "influences":         "ASSOCIATES",
        "shapes":             "ASSOCIATES",
        "reduces":            "ASSOCIATES",
        "correlates_with":    "ASSOCIATES",
        "is_associated_with": "ASSOCIATES",
    },

    # ---- LLM call sizing ----
    # Partition outputs scale with shard_size; 2048 truncates in the
    # middle of a 500-prop shard's group list. 8192 fits comfortably
    # for the closed-vocabulary Anthropic Haiku models under
    # step_b_mode="exhaustive". Under step_b_mode="pivot" the Step A
    # response carries snippet metadata per member, ballooning the
    # response size by ~10x — 8192 tokens truncates around index ~335
    # of a ~477-prop shard. 32768 is safely above the observed ceiling.
    "max_tokens_partition": 32768,
    # Bumped from 4096: the exhaustive-coverage Step B prompt asks the
    # LLM to consider every (member, relation, probe, direction) cell,
    # so the asserted-true list can grow much larger than under the
    # earlier sparse-enumeration framing.
    "max_tokens_expansion": 8192,
    "batch_size":           1,

    # ---- quotient_report.py ----
    # Set these before running `python -m scripts.quotient_report`.
    # Paths are interpreted relative to the project root
    # (Democritus_OpenAI/).
    "report_baseline_dir":  "runs/smoke_test",
    "report_treatment_dir": "runs/smoke_test_quotient",
    "report_output_path":   "runs/quotient_report.md",

    # =================================================================
    # Prompt templates
    # =================================================================
    #
    # Every LLM-facing string used by scripts/indiscernibility_quotient.py
    # lives here. The module reads CONFIG["prompts"][key] at first use;
    # there are no prompt strings hard-coded in the module.
    #
    # Each template is a Python str.format() template — placeholders are
    # listed in the comment above the template. The required JSON output
    # shape and the parser that consumes it are listed too, since
    # changing the wording without preserving the response schema will
    # break parsing.
    #
    # Editing wording is safe; editing or omitting placeholders is not.
    # The module validates at import time that every active template's
    # placeholders match the keys its builder will pass (see
    # `_validate_prompt_placeholders` in indiscernibility_quotient.py).
    "prompts": {

        # -- Step A: candidate similarity sets --------------------------
        #
        # Used by:    partition_propositions (Step A)
        #             when CONFIG["step_b_mode"] == "exhaustive".
        # Builder:    _build_similarity_sets_prompt
        # Parser:     _parse_similarity_sets_response
        # Placeholders: {max_class_size}, {numbered_list}
        # Response shape: {"sets": [[i, j, ...], [k, l, ...], ...]}
        #
        # Recall-biased: under-grouping is permanent, over-grouping is
        # cheap (Step C's strict identity check filters spurious sets).
        # Names five concrete grouping criteria so the LLM does not fall
        # back to its default "paraphrase" intuition.
        "step_a_exhaustive": """\
You will scan a numbered list of causal propositions and identify
SETS of propositions that are RELATED ENOUGH to investigate whether
they assert the same underlying causal claim. The downstream pipeline
performs a strict identity check on each set you produce, so
over-grouping is cheap (a bad set is filtered out); under-grouping is
permanent (a missed pair will never be merged).

Be generous. Group propositions that share any of the following:
  (a) surface paraphrase --- same claim, different wording
  (b) same subject or object referent in different phrasings
      (e.g. "monsoon intensity" and "rising monsoon strength")
  (c) same causal mechanism described from different angles
  (d) specific and general versions of the same claim
      (one is a more specific instance of the other)
  (e) same cause and same effect, even if the relation verb differs
      in strength or direction emphasis

Rules:
- Each set must have at least 2 members.
- A proposition MAY appear in multiple sets.
- Set size must not exceed {max_class_size}.
- A proposition that fits no set may be left out.

Propositions (numbered):
{numbered_list}

Return JSON only, no commentary, in this exact shape:
{{"sets": [[i, j, ...], [k, l, ...], ...]}}
""",

        # -- Step A pivot mode -----------------------------------------
        #
        # Used by:    partition_propositions (Step A)
        #             when CONFIG["step_b_mode"] == "pivot".
        # Builder:    _build_similarity_sets_prompt_pivot
        # Parser:     _parse_similarity_sets_response_pivot
        #             (validates each member's snippet via _snippet_matches
        #              and drops members whose snippet does not appear in
        #              the indexed proposition).
        # Placeholders: {max_class_size}, {numbered_list}
        # Response shape: {"sets": [
        #   [{"i": int, "snippet": str}, ...],
        #   ...
        # ]}
        #
        # The first member of each set is the LLM-designated rep used as
        # the anchor for Step B Phase 1 / Phase 2. Snippet grounding
        # catches LLM index drift (the LLM lost track of which integer
        # maps to which proposition).
        "step_a_pivot": """\
You will scan a numbered list of causal propositions and identify
SETS of propositions that are RELATED ENOUGH to investigate whether
they assert the same underlying causal claim. The downstream pipeline
performs a strict identity check on each set you produce, so
over-grouping is cheap (a bad set is filtered out); under-grouping is
permanent (a missed pair will never be merged).

Be generous. Group propositions that share any of the following:
  (a) surface paraphrase --- same claim, different wording
  (b) same subject or object referent in different phrasings
      (e.g. "monsoon intensity" and "rising monsoon strength")
  (c) same causal mechanism described from different angles
  (d) specific and general versions of the same claim
      (one is a more specific instance of the other)
  (e) same cause and same effect, even if the relation verb differs
      in strength or direction emphasis

Rules:
- Each set must have at least 2 members.
- A proposition MAY appear in multiple sets.
- Set size must not exceed {max_class_size}.
- A proposition that fits no set may be left out.

For each member of each set, emit BOTH:
  - "i": the integer index of the proposition (as numbered below).
  - "snippet": the first ~5 words of the proposition copied exactly
    from the input. The downstream parser uses this as a grounding
    check; if your snippet does not appear in the proposition at the
    given index, the member is dropped.

The FIRST member of each set is the SET REPRESENTATIVE — pick the
member that is the most prototypical or canonical phrasing of the
shared claim. The downstream identity check is unaffected by this
choice (any reasonable representative is fine), but the rep is used
as an anchor for cross-member edge comparison.

Propositions (numbered):
{numbered_list}

Return JSON only, no commentary, in this exact shape:
{{"sets": [
  [
    {{"i": 0, "snippet": "first few words"}},
    {{"i": 5, "snippet": "first few words"}}
  ],
  [
    {{"i": 11, "snippet": "..."}},
    {{"i": 19, "snippet": "..."}}
  ]
]}}
""",

        # -- Step B exhaustive mode ------------------------------------
        #
        # Used by:    _expand_exhaustive (Step B)
        #             when CONFIG["step_b_mode"] == "exhaustive".
        # Builder:    _build_expansion_prompt
        # Parser:     _parse_expansion_response
        # Placeholders: {numbered_members_with_context}, {numbered_probes},
        #               {rel_list}
        # Response shape: {"asserted": [
        #   {"m": int, "r": str, "z": int, "d": "out"|"in"},
        #   ...
        # ]}
        #
        # One LLM call per candidate class. Asks the LLM to enumerate
        # every (member, relation, probe, direction) cell; cells the
        # LLM does not emit are taken to be false. Becomes unreliable
        # for class sizes > ~5 — that is the reason pivot mode exists.
        "step_b_exhaustive": """\
You will check causal relations within a fixed closed vocabulary
between members of a set of similar propositions and a list of probe
propositions.

The set members are propositions that may be near-paraphrases of one
another. Each member is shown WITH the upstream context that produced
it (the topic path, the originating question, and the sentence the
member was extracted from), so that the underlying claim is
unambiguous despite differences in surface phrasing. The probes are
other propositions in the causal graph. For each (member, relation,
probe, direction) cell, decide whether the causal claim holds in
general knowledge, using the upstream context to disambiguate the
member's intended scope.

Set members (numbered, with upstream context):
{numbered_members_with_context}

Probe propositions (numbered):
{numbered_probes}

Relations: {rel_list}

Be EXHAUSTIVE: consider every (member, relation, probe, direction)
combination. Each member is a separate question — if member 0 has a
relation to a probe, do not silently omit member 1: answer the
analogous cell for member 1 explicitly. The judgment is about what
the LLM knows in general, not about whether the relation has been
previously stated.

Direction "out" means (member, relation, probe); direction "in" means
(probe, relation, member). Do NOT emit cells outside the given
vocabulary.

For each cell that holds, emit one JSON object with the keys m, r, z,
d. Cells you do not emit are taken to be false.

Return JSON only, no commentary, in this exact shape:
{{"asserted": [
  {{"m": 0, "r": "causes", "z": 3, "d": "out"}},
  {{"m": 1, "r": "leads_to", "z": 7, "d": "in"}}
]}}
""",

        # -- Step B pivot mode, Phase 1: lift onto rep ------------------
        #
        # Used by:    _phase1_lift
        #             when CONFIG["step_b_mode"] == "pivot".
        # Builder:    inlined in _phase1_lift.
        # Parser:     _parse_pivot_phase_response
        # Placeholders: {rep_block}, {numbered_members_with_edges},
        #               {rel_list}
        # Response shape: {"asserted": [
        #   {"m": int, "e": int, "holds": bool},
        #   ...
        # ]}
        #
        # For each (non-rep member, member-edge) cell, decides whether
        # the analogous edge attaches to the rep too. Affirmative cells
        # become expansion edges anchored on the rep. The judgment is
        # about the rep itself — the prompt is deliberately neutral on
        # whether rep and member are equivalent.
        "step_b_phase1_lift": """\
You will check whether causal edges currently asserted on each NON-REP
member of a near-paraphrase set ALSO hold for the REP. If yes, the
edge is added to the rep's edge set as an LLM-asserted expansion edge.

The rep and the non-rep members may be paraphrases of one another. Use
each member's upstream context (topic path, originating question, and
the sentence the member was extracted from) to disambiguate intended
scope.

Each edge is a tuple (direction, relation, neighbour). "out" means
(member, relation, neighbour); "in" means (neighbour, relation,
member).

REP (with upstream context):
{rep_block}

NON-REP MEMBERS, each with upstream context AND its currently-asserted
causal edges (numbered locally per member):
{numbered_members_with_edges}

Closed relation vocabulary: {rel_list}

For each (member, edge) cell, decide whether the analogous edge holds
for the REP --- that is, does the same (direction, relation,
neighbour) edge attach to the rep's claim too? The judgment is about
the rep itself, not about whether the rep and the member are
equivalent.

Return JSON only, no commentary, in this exact shape:
{{"asserted": [
  {{"m": 0, "e": 0, "holds": true}},
  {{"m": 0, "e": 1, "holds": false}},
  {{"m": 1, "e": 0, "holds": true}}
]}}

`m` indexes the non-rep member list; `e` indexes that member's edge
list. Cells you omit are taken as `holds: false`.
""",

        # -- Step B pivot mode, Phase 2: distribute onto non-reps -------
        #
        # Used by:    _phase2_distribute
        #             when CONFIG["step_b_mode"] == "pivot".
        # Builder:    inlined in _phase2_distribute.
        # Parser:     _parse_pivot_phase_response
        # Placeholders: {rep_block_with_edges},
        #               {numbered_members_with_context}, {rel_list}
        # Response shape: {"asserted": [
        #   {"m": int, "e": int, "holds": bool},
        #   ...
        # ]}
        #
        # For each (non-rep member, rep-edge) cell, decides whether the
        # analogous edge attaches to that non-rep member. Affirmative
        # cells become expansion edges anchored on the non-rep member.
        # Run AFTER Phase 1, so the rep's edge list here is the union
        # of extraction edges and Phase-1-lifted edges.
        "step_b_phase2_distribute": """\
You will check whether each causal edge currently asserted on the REP
ALSO holds for each NON-REP member of a near-paraphrase set. If yes,
the edge is added to that member's edge set as an LLM-asserted
expansion edge.

Each edge is a tuple (direction, relation, neighbour). "out" means
(member, relation, neighbour); "in" means (neighbour, relation,
member).

REP (with upstream context AND its currently-asserted causal edges,
numbered):
{rep_block_with_edges}

NON-REP MEMBERS, each with upstream context (numbered):
{numbered_members_with_context}

Closed relation vocabulary: {rel_list}

For each (member, rep-edge) cell, decide whether the analogous edge
holds for that member.

Return JSON only, no commentary, in this exact shape:
{{"asserted": [
  {{"m": 0, "e": 0, "holds": true}},
  {{"m": 1, "e": 0, "holds": true}},
  {{"m": 0, "e": 2, "holds": false}}
]}}

`m` indexes the non-rep member list; `e` indexes the rep's edge list.
Cells you omit are taken as `holds: false`.
""",

        # -- Retry suffix ----------------------------------------------
        #
        # Appended to any prompt whose first response failed to parse.
        # Used by:    Step A (both modes), Step B exhaustive, Phase 1,
        #             Phase 2 — every site that issues an LLM call
        #             expecting JSON.
        # Helper:     _retry_prompt(original_prompt, prior_response)
        # Placeholders: {prior_response}
        #
        # The original prompt is concatenated as a prefix; this template
        # is the suffix only.
        "retry_suffix": """\

Your previous output could not be parsed:
{prior_response}

Return ONLY the JSON object specified above.""",

        # -- Step A legacy v6 (INACTIVE; documentation only) ------------
        #
        # The v6 conservative wording, kept verbatim for reference.
        # Replaced by step_a_exhaustive / step_a_pivot in v7 (recall-
        # biased). Reading: the three "permission-to-be-lazy" sentences
        # ("Precision matters; recall does not.", "Most propositions
        # will not [be in any set].", "Stop emitting sets once no
        # further confident groupings remain.") are exactly what we
        # removed when flipping the bias.
        #
        # Not referenced by any builder. Present so the design history
        # is auditable from a single file.
        "step_a_legacy_v6": """\
You will scan a numbered list of causal propositions and identify SETS
OF NEAR-PARAPHRASES: propositions that assert the same underlying
causal claim despite differences in surface phrasing.

Rules:
- Only emit a set when you are CONFIDENT the propositions in it are
  saying the same thing. Precision matters; recall does not.
- A proposition does NOT need to belong to any set. Most propositions
  will not. That is correct.
- Each set must have at least 2 members.
- A proposition MAY appear in multiple sets if it is genuinely a
  near-paraphrase of multiple distinct claims.
- Set size must not exceed {max_class_size}.
- Stop emitting sets once no further confident groupings remain. An
  empty list is a valid answer if no near-paraphrases exist.

Propositions (numbered):
{numbered_list}

Return JSON only, no commentary, in this exact shape:
{{"sets": [[i, j, ...], [k, l, ...], ...]}}
""",
    },
}

#!/usr/bin/env python3
"""
causal_verbs.py
---------------

Single source of truth for the causal-verb vocabulary used across
Modules 2, 3, and 4. Before consolidation, the verb list appeared in
seven separate places (two prompt strings, two ranking-stopword sets,
one keyword list, and one regex pattern table that was defined twice
within the same file). This allowed silent drift; for example, the
question-side stopword set was missing ``affects`` while the statement-
side set included it, and the extractor's regex table was identically
redefined in two places, so any future edit to only one of them would
have produced a hard-to-diagnose bug.

Designer switch
~~~~~~~~~~~~~~~

``USE_EXTENDED_VERB_SET`` (top of file) selects between two
configurations:

  - ``False`` (default, original behavior): the prompts in Modules 2
    and 3 request the six base verbs; the extractor in Module 4
    additionally admits four extras (``shapes``, ``contributes to``,
    ``correlates with``, ``is associated with``) as a permissiveness
    layer for LLM phrasings that drift from the prompt set.

  - ``True``: all ten verbs are unified. The prompts request all ten,
    the keyword list covers their inflections, and the stopword set
    covers their head tokens. The extractor still admits all ten, but
    EXTRACTOR_ONLY_VERBS becomes empty because every verb is now in
    PROMPT_VERBS.

Caution: when ``USE_EXTENDED_VERB_SET`` is True, the prompts request
``correlates with`` and ``is associated with``, both of which describe
*statistical association* rather than *causation*. Asking the LLM to
use these will admit non-causal claims into the database. Use the
extended set deliberately, e.g. for ablation studies that compare
causal-only against association-tolerant extraction.
"""

from typing import Dict, FrozenSet, Tuple


# ---------------------------------------------------------------------
# Designer switch
# ---------------------------------------------------------------------

USE_EXTENDED_VERB_SET: bool = True


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------

# The six surface forms originally used by the prompts in Modules 2
# and 3. Order matches the original ``REL_PATTERNS`` insertion order,
# preserving the extractor's first-match priority.
_BASE_VERBS: Tuple[str, ...] = (
    "causes",
    "leads to",
    "increases",
    "reduces",
    "affects",
    "influences",
)

# Four surface forms admitted by the extractor before consolidation.
# When USE_EXTENDED_VERB_SET is False, these remain extractor-only;
# when True, they are appended to PROMPT_VERBS and treated identically.
_EXTRA_VERBS: Tuple[str, ...] = (
    "shapes",
    "contributes to",
    "correlates with",
    "is associated with",
)

# Inflected forms keyed by surface form, in (stem, third-person-singular,
# past) order. For the passive idiom ``is associated with`` the auxiliary
# verb is inflected and the participle is fixed:
# (be associated with, is associated with, was associated with).
_INFLECTIONS: Dict[str, Tuple[str, str, str]] = {
    "causes":             ("cause",              "causes",             "caused"),
    "leads to":           ("lead to",            "leads to",           "led to"),
    "increases":          ("increase",           "increases",          "increased"),
    "reduces":            ("reduce",             "reduces",            "reduced"),
    "affects":            ("affect",             "affects",            "affected"),
    "influences":         ("influence",          "influences",         "influenced"),
    "shapes":             ("shape",              "shapes",             "shaped"),
    "contributes to":     ("contribute to",      "contributes to",     "contributed to"),
    "correlates with":    ("correlate with",     "correlates with",    "correlated with"),
    "is associated with": ("be associated with", "is associated with", "was associated with"),
}


# ---------------------------------------------------------------------
# Public surfaces (computed at import time from the switch above)
# ---------------------------------------------------------------------

PROMPT_VERBS: Tuple[str, ...] = (
    _BASE_VERBS + _EXTRA_VERBS if USE_EXTENDED_VERB_SET else _BASE_VERBS
)

EXTRACTOR_ONLY_VERBS: Tuple[str, ...] = (
    () if USE_EXTENDED_VERB_SET else _EXTRA_VERBS
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def relation_key(surface: str) -> str:
    """Map a surface form to a canonical relation key.

    Example: ``"leads to" -> "leads_to"``. The result is the dict key
    used in ``REL_PATTERNS`` and stored in the ``rel`` field of each
    extracted triple.
    """
    return surface.replace(" ", "_")


def prompt_verb_fragment() -> str:
    """Comma-separated PROMPT_VERBS for direct interpolation into LLM prompts."""
    return ", ".join(PROMPT_VERBS)


def causal_keywords() -> Tuple[str, ...]:
    """Flat tuple of every inflected form of every PROMPT_VERB, suitable for
    substring containment tests (Module 3 uses
    ``any(kw in sentence_lower for kw in CAUSAL_KEYWORDS)``).

    Order is PROMPT_VERBS order, with each verb's inflections appended in
    (stem, present, past) order. With ``USE_EXTENDED_VERB_SET = False`` the
    result is bit-identical to the pre-consolidation ``CAUSAL_KEYWORDS``
    list of 18 forms.
    """
    return tuple(form for verb in PROMPT_VERBS for form in _INFLECTIONS[verb])


def verb_stopwords() -> FrozenSet[str]:
    """Head tokens of all causal_keywords(), for use as ranking stopwords by
    Modules 2 and 3.

    Multi-word forms (``leads to``, ``is associated with``, ...) contribute
    only their head token because the upstream tokenizer in both consumers
    splits on non-alphanumerics and drops tokens of length <= 2; the
    trailing ``to`` / ``with`` / ``associated`` are either short enough to
    be filtered or are themselves admitted as content tokens.
    """
    return frozenset(form.split()[0] for form in causal_keywords())


def rel_patterns(include_extractor_only: bool = True) -> Dict[str, str]:
    """Build the REL_PATTERNS dictionary consumed by Module 4.

    Each value is a regex source string of the form
    ``(.+?)\\s+<surface>\\s+(.+)``. The surface is interpolated
    literally; the current vocabulary contains only letters and
    spaces, so no metacharacter handling is needed. If a future verb
    includes a regex metacharacter (``+``, ``.``, ``?``, ``(``, ...),
    that case must be handled deliberately at the call site rather
    than papered over here.

    Parameters
    ----------
    include_extractor_only
        When True (default), the four EXTRACTOR_ONLY_VERBS are appended
        after PROMPT_VERBS, preserving the extractor's pre-consolidation
        first-match order. When ``USE_EXTENDED_VERB_SET`` is True,
        EXTRACTOR_ONLY_VERBS is empty and this argument has no effect.
    """
    surfaces: Tuple[str, ...] = PROMPT_VERBS + (
        EXTRACTOR_ONLY_VERBS if include_extractor_only else ()
    )
    return {
        relation_key(s): rf"(.+?)\s+{s}\s+(.+)"
        for s in surfaces
    }

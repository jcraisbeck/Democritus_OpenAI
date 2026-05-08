#!/usr/bin/env python3
"""
render_viewer.py
----------------

Generate the CLIFF-style relational manifold viewer for a Democritus
run directory without invoking the full CLIFF batch pipeline.

Reads one Democritus run output (the directory produced by
``pipelines.pipeline_llm``) plus the ``root_topics.txt`` that produced
it, and writes a self-contained HTML viewer next to the manifold PNG.

The output is feature-equivalent to CLIFF's
``relational_manifold_viewer.html`` *minus* the LCM gallery (LCMs
require a separate scoring pass not run here). The HTML/CSS/JS
template is vendored from
``CLIFF_CatAgi/functorflow_v3/democritus_batch_agentic.py``,
``_render_manifold_viewer_html`` (roughly lines 2014-2342) so this
script does not depend on CLIFF being importable. To pull a future
CLIFF improvement, re-copy the template body in ``build_html`` below.

Usage
-----
Edit the CONFIG block below, then run::

    python render_viewer.py

Side effects: writes the HTML file at OUTPUT_PATH and up to four
sidecar files next to it (kept out of the HTML so the document stays
small):

    groups_data.json      canonical record of the "All grouped
                          propositions" panel: compression stats and
                          the per-group (id, representative, members)
                          rows. Written when the run produced any
                          multi-member groups.
    groups_data.js        same payload wrapped as
                          ``window.GROUPS_DATA = <json>;``.
    relations_data.json   canonical record of the "All causal
                          relationships" panel: per-row subj, obj,
                          display relation, domain, edge-source
                          class, and any merged-from member groups.
                          Written when the run has any triples.
    relations_data.js     same payload wrapped as
                          ``window.RELATIONS_DATA = <json>;``.

The ``.js`` wrappers exist because browsers block ``fetch()`` from
``file://`` URLs, but ``<script src>`` works there. The ``.json``
files are the format other tools should consume.

Does not modify any other artifact in the run directory.
"""

import html
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT: Path = Path(__file__).resolve().parent


# =====================================================================
# CONFIG — edit these, then run ``python render_viewer.py``.
# =====================================================================

# The Democritus run directory to render. Must contain:
#   - viz/relational_manifold_2d.png
#   - viz/relational_manifold_labels.json
#   - relational_triples.jsonl
RUN_DIR: str = "runs/smoke_test"

# Root topics file (one topic per non-blank line). Default is the path
# produced by ``scripts.document_topic_discovery``.
ROOT_TOPICS_PATH: str = "configs/root_topics.txt"

# Output HTML path. Must be in the same directory as the manifold PNG
# so the relative ``<img src="relational_manifold_2d.png">`` resolves.
OUTPUT_PATH: str = "runs/smoke_test/viz/manifold_viewer.html"

# Title shown in the hero card. If empty, defaults to RUN_DIR's name.
TITLE: str = ""

# Number of triples shown in the "Key Evidence Labels" panel. The
# script reads the first TOP_TRIPLE_COUNT *unique* (subj, rel, obj)
# rows in file order — there is no scoring step here, so "first" means
# first in the JSONL, not "best".
TOP_TRIPLE_COUNT: int = 4


# =====================================================================
# Helpers
# =====================================================================

def _esc(value: object) -> str:
    """HTML-escape a value's str representation."""
    return html.escape(str(value))


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce to float; return default for non-numeric inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _display_rel(record: Dict[str, Any]) -> str:
    """Choose the relation label to show in the viewer.

    Prefers the original verb(s) preserved in `original_rels` (a list
    populated by indiscernibility_quotient when relation softening is
    in effect) over the canonical class in `rel`. Falls back to `rel`
    for records produced by the unmodified pipeline (Module 4 output)
    which has no `original_rels` field.

    Multiple original verbs are joined with " / " — this happens when
    the relation-equivalence map collapsed several raw verbs onto one
    canonical class for the same (subj, obj) pair."""
    orig = record.get("original_rels")
    if isinstance(orig, list) and orig:
        return " / ".join(orig)
    return str(record.get("rel") or "")


def _edge_source_class(record: Dict[str, Any]) -> str:
    """CSS class for a relations-panel row based on its
    ``edge_source`` field. Rows that carry extraction provenance
    (extraction-only or mixed) get ``rel-row-has-extraction``; pure
    expansion rows get ``rel-row-expansion-only``, which the default
    CSS hides until the user toggles them on."""
    src = record.get("edge_source")
    if isinstance(src, list):
        if "extraction" in src:
            return "rel-row-has-extraction"
        return "rel-row-expansion-only"
    if src == "expansion":
        return "rel-row-expansion-only"
    # Missing or "extraction" string → treat as extraction.
    return "rel-row-has-extraction"


def _member_groups(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-row "merged from..." metadata for a triple whose subj or
    obj was collapsed from multiple source propositions. Returns one
    dict per merged side (zero, one, or two entries) with keys
    ``side`` ("subj" or "obj"), ``group_id``, and ``members``.
    Singleton sides are omitted since they carry no merge info."""
    out: List[Dict[str, Any]] = []
    for side in ("subj", "obj"):
        members = record.get(f"members_{side}") or []
        if len(members) <= 1:
            continue
        out.append({
            "side":     side,
            "group_id": record.get(f"group_id_{side}") or "",
            "members":  list(members),
        })
    return out


def read_root_topics(path: Path) -> List[str]:
    """Read root_topics.txt; one topic per non-blank line."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_top_triples(jsonl_path: Path, n: int) -> List[Dict[str, Any]]:
    """Read the first ``n`` unique ``(subj, rel, obj)`` rows from the
    relational_triples JSONL, preserving file order. Duplicates beyond
    the first occurrence are skipped silently."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        key = (item.get("subj"), item.get("rel"), item.get("obj"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= n:
            break
    return out


def read_all_triples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read every row of the relational_triples JSONL, in file order.
    Used to populate the collapsible "all relationships" panel; no
    deduplication is applied so duplicate extractions remain visible."""
    out: List[Dict[str, Any]] = []
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def read_hover_clusters(json_path: Path) -> List[Dict[str, Any]]:
    """Load the cluster list from relational_manifold_labels.json.
    Each cluster is expected to carry ``label``, ``sample_labels``,
    ``point_count``, ``x_norm``, ``y_norm``, ``radius_norm`` (all
    optional; missing values fall back to renderer defaults)."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return list(payload.get("clusters", []))


def extract_groups(all_triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Walk the grouped triples and return one record per equivalence
    class whose member count exceeds 1.

    Each returned record:

        {"group_id":       "g_000",
         "representative": "<longest subj/obj string seen for the group>",
         "members":        ["...", "..."]}

    Triples without ``group_id_subj`` / ``group_id_obj`` (i.e. raw
    extraction output from a non-quotient run) are skipped silently,
    so on a raw run the function returns an empty list and the viewer
    suppresses the groups section. Returned list is sorted by
    group_id for stable rendering. Singletons are omitted: they carry
    no merge information for the reader.

    The chosen representative is the longest member string seen
    across all rows for the group, matching the quotient module's
    default representative-selection rule."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in all_triples:
        for side in ("subj", "obj"):
            gid = item.get(f"group_id_{side}")
            if not gid:
                continue
            members = item.get(f"members_{side}") or []
            candidate_rep = item.get(side) or ""
            if gid not in by_id:
                by_id[gid] = {
                    "group_id": gid,
                    "representative": candidate_rep,
                    "members_set": set(str(m) for m in members),
                }
                continue
            entry = by_id[gid]
            entry["members_set"].update(str(m) for m in members)
            if len(candidate_rep) > len(entry["representative"]):
                entry["representative"] = candidate_rep
    out: List[Dict[str, Any]] = []
    for gid in sorted(by_id):
        entry = by_id[gid]
        members = sorted(entry["members_set"])
        if len(members) <= 1:
            continue
        out.append({
            "group_id": gid,
            "representative": entry["representative"],
            "members": members,
        })
    return out


def compute_compression_stats(
    all_triples: List[Dict[str, Any]],
    all_groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Vocabulary-level compression metrics for the
    "All grouped propositions" panel.

    Returns a dict with keys:

        propositions_before        : |V|, the pre-quotient proposition
            count. Union of all `members_subj` / `members_obj` strings
            across triples. Raw-extraction rows that lack the
            `members_*` field contribute their own subj / obj strings
            as singleton pre-quotient members.
        classes_after              : |V/~|, the post-quotient class
            count. Union of distinct `group_id_subj` /
            `group_id_obj` values. Raw rows contribute distinct
            subj / obj strings as their own singleton classes.
        compression_pct            : 100 · (1 − classes_after /
            propositions_before). Zero when |V| = 0.
        n_compressed_groups        : len(all_groups). Number of
            multi-member classes (singletons excluded).
        avg_compressed_group_size  : mean of len(members) over
            `all_groups`. Singletons excluded since they carry no
            compression. Zero when there are no multi-member groups."""
    pre_members: set = set()
    post_classes: set = set()
    for item in all_triples:
        for side in ("subj", "obj"):
            members = item.get(f"members_{side}") or []
            if members:
                for m in members:
                    pre_members.add(str(m))
            else:
                rep = item.get(side) or ""
                if rep:
                    pre_members.add(rep)
            gid = item.get(f"group_id_{side}")
            if gid:
                post_classes.add(("g", str(gid)))
            else:
                rep = item.get(side) or ""
                if rep:
                    post_classes.add(("raw", rep))
    n_before = len(pre_members)
    n_after = len(post_classes)
    compression_pct = 100.0 * (1.0 - n_after / n_before) if n_before > 0 else 0.0
    if all_groups:
        avg_compressed_group_size = sum(len(g["members"]) for g in all_groups) / len(all_groups)
    else:
        avg_compressed_group_size = 0.0
    return {
        "propositions_before":       n_before,
        "classes_after":             n_after,
        "compression_pct":           compression_pct,
        "n_compressed_groups":       len(all_groups),
        "avg_compressed_group_size": avg_compressed_group_size,
    }


# =====================================================================
# Renderer
#
# Vendored from CLIFF democritus_batch_agentic.py
# (_render_manifold_viewer_html). The LCM gallery section is replaced
# with a static placeholder since that path is not generated here.
# =====================================================================

def build_html(
    *,
    title: str,
    manifold_filename: str,
    root_topics: List[str],
    top_triples: List[Dict[str, Any]],
    all_triples: List[Dict[str, Any]],
    hover_clusters: List[Dict[str, Any]],
    all_groups: List[Dict[str, Any]],
    compression_stats: Dict[str, Any],
) -> str:
    """Render the relational manifold viewer HTML as a single string.

    Returns a self-contained document with embedded CSS and a small
    hover-interaction script. ``manifold_filename`` is the basename of
    the PNG (e.g. ``relational_manifold_2d.png``); the HTML must be
    written into the same directory as the PNG so the relative
    ``<img src=...>`` resolves.

    An empty ``hover_clusters`` falls back to a help note. An empty
    ``top_triples`` falls back to an "Relational triples will appear
    here..." panel. An empty ``all_groups`` (raw extraction run) causes
    the "All grouped propositions" section to be omitted entirely.

    The two large list panels — "All causal relationships" and
    "All grouped propositions" — are rendered in the browser, not
    here. The HTML contains only their chrome (summary text, stats,
    toggle, empty ``<ol>`` placeholders) plus ``<script src=...>``
    tags pointing at ``relations_data.js`` and ``groups_data.js``.
    The rows themselves come from ``window.RELATIONS_DATA`` and
    ``window.GROUPS_DATA``. The caller (``main``) is responsible for
    writing the sidecar ``.js`` and matching ``.json`` files next to
    the HTML.
    """
    esc = _esc

    topic_markup = "".join(
        f'<span class="chip" data-topic-label="{esc(topic)}">{esc(topic)}</span>'
        for topic in root_topics[:10]
    )
    triple_markup = "".join(
        '<article class="mini-card">'
        f'<div class="mini-title">{esc(item.get("subj") or "")} <span class="arrow">→</span> {esc(item.get("obj") or "")}</div>'
        f'<p>{esc(item.get("statement") or _display_rel(item))}</p>'
        "</article>"
        for item in top_triples[:4]
    )
    hover_markup = "".join(
        '<button class="manifold-hotspot" type="button"'
        f' style="left: {100.0 * _safe_float(item.get("x_norm"), 0.5):.2f}%; top: {100.0 * _safe_float(item.get("y_norm"), 0.5):.2f}%; '
        f'width: {200.0 * _safe_float(item.get("radius_norm"), 0.08):.2f}%; height: {200.0 * _safe_float(item.get("radius_norm"), 0.08):.2f}%;"'
        f' data-label="{esc(item.get("label") or "")}"'
        f' data-topic="{esc(item.get("matched_topic") or "")}"'
        f' data-size="{esc(item.get("point_count") or 0)}"'
        f' data-samples="{esc(" | ".join(str(part) for part in list(item.get("sample_labels") or [])[:4]))}">'
        '<span class="sr-only">Show manifold label</span>'
        "</button>"
        for item in hover_clusters
    )
    hover_help = (
        '<div class="viewer-note">Hover the manifold to surface the nearest recovered cluster label and its matched topic chip.</div>'
        if hover_clusters
        else '<div class="viewer-note">Cluster hover labels will appear automatically when manifold metadata is available.</div>'
    )
    n_expansion_only = sum(
        1 for t in all_triples
        if _edge_source_class(t) == "rel-row-expansion-only"
    )
    n_with_extraction = len(all_triples) - n_expansion_only

    if n_expansion_only > 0:
        toggle_markup = (
            '<div class="relations-toggle">'
            '<button type="button" class="toggle-expansion" data-state="off"'
            f' data-n-hidden="{n_expansion_only}"'
            f' data-n-with-ext="{n_with_extraction}"'
            f' data-n-total="{len(all_triples)}">'
            f"Show {n_expansion_only} LLM-asserted expansion edges"
            "</button>"
            "</div>"
        )
        summary_text = (
            f"<strong>All causal relationships</strong> &middot; "
            f"{n_with_extraction} document-grounded "
            f"(of {len(all_triples)} total; "
            f"{n_expansion_only} pure-expansion hidden)"
        )
    else:
        toggle_markup = ""
        summary_text = (
            f"<strong>All causal relationships</strong> &middot; "
            f"{len(all_triples)} total"
        )
    # Relations rows are rendered in the browser from
    # window.RELATIONS_DATA (see relations_data.js, written by main()).
    # Same pattern as the groups panel: chrome stays inline so the
    # summary count and toggle are meaningful before JS runs; the rows
    # live in the sidecar.
    relations_markup = (
        '<section class="panel relations-panel">'
        "<details>"
        f"<summary>{summary_text}</summary>"
        f"{toggle_markup}"
        '<ol class="rel-list" id="relations-list"></ol>'
        "</details>"
        "</section>"
        '<script src="relations_data.js"></script>'
    ) if all_triples else ""

    # Group rows are rendered in the browser from window.GROUPS_DATA
    # (see groups_data.js, written by main()). We emit only the
    # section's chrome here. The summary count and stats block use
    # values known at render time so the header is meaningful even
    # before the sidecar script has been parsed.
    cs = compression_stats
    groups_stats_markup = (
        '<div class="groups-stats">'
        f'<div><strong>|V|</strong> {cs["propositions_before"]} '
        f'→ <strong>|V/~|</strong> {cs["classes_after"]} '
        f'· <strong>compression</strong> {cs["compression_pct"]:.1f}%</div>'
        f'<div><strong>{cs["n_compressed_groups"]}</strong> multi-member classes '
        f'· average compressed-group size <strong>{cs["avg_compressed_group_size"]:.2f}</strong></div>'
        "</div>"
    )
    groups_markup = (
        '<section class="panel groups-panel">'
        "<details>"
        f"<summary><strong>All grouped propositions</strong> &middot; {len(all_groups)} total</summary>"
        f"{groups_stats_markup}"
        '<ol class="rel-list" id="groups-list"></ol>'
        "</details>"
        "</section>"
        '<script src="groups_data.js"></script>'
    ) if all_groups else ""

    image_markup = (
        '<div class="viewer-grid">'
        '<aside class="hover-rail">'
        '<div class="hover-card" id="hover-card">'
        '<div class="hover-eyebrow">Hovered Cluster</div>'
        '<div class="hover-title" id="hover-title">Move over the manifold</div>'
        '<div class="hover-meta" id="hover-meta">Recovered local labels will appear here.</div>'
        "</div>"
        "</aside>"
        '<div class="manifold-stage">'
        f'<img src="{esc(manifold_filename)}" alt="{esc(title)} relational manifold" loading="eager">'
        f"{hover_markup}"
        "</div>"
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manifold View · {esc(title)}</title>
  <style>
    :root {{
      --ink: #142433;
      --muted: #5a7184;
      --paper: #f5efe5;
      --card: rgba(255, 252, 246, 0.97);
      --line: #d8ccbc;
      --accent: #0f766e;
      --shadow: 0 18px 44px rgba(20, 36, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 28%),
        linear-gradient(180deg, #fbf6ee 0%, #eee4d6 100%);
    }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 44px; }}
    .hero, .viewer, .panel {{
      background: var(--card);
      border: 1px solid rgba(216, 204, 188, 0.96);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 26px; margin-bottom: 20px; }}
    .eyebrow {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.16em; color: var(--muted); }}
    h1 {{ margin: 10px 0 8px; font-size: clamp(32px, 4vw, 54px); line-height: 0.98; }}
    .hero p {{ margin: 0; color: var(--muted); font-size: 17px; line-height: 1.65; max-width: 920px; }}
    .layout {{ display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 18px; }}
    .viewer {{ padding: 18px; }}
    .viewer-grid {{
      display: grid;
      grid-template-columns: minmax(220px, 0.34fr) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .hover-rail {{ min-width: 0; }}
    .manifold-stage {{ position: relative; }}
    .viewer img {{ width: 100%; border-radius: 22px; display: block; background: white; }}
    .viewer-note {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }}
    .manifold-hotspot {{
      position: absolute;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 2px solid rgba(15, 118, 110, 0.18);
      background: rgba(15, 118, 110, 0.06);
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }}
    .manifold-hotspot:hover,
    .manifold-hotspot:focus-visible,
    .manifold-hotspot.is-active {{
      border-color: rgba(15, 118, 110, 0.9);
      background: rgba(15, 118, 110, 0.18);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.14);
      outline: none;
    }}
    .hover-card {{
      position: sticky;
      top: 0;
      min-height: 150px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(20, 36, 51, 0.94);
      color: white;
      box-shadow: 0 12px 32px rgba(20, 36, 51, 0.2);
      backdrop-filter: blur(8px);
    }}
    .hover-eyebrow {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      opacity: 0.72;
    }}
    .hover-title {{ margin-top: 6px; font-size: 20px; line-height: 1.25; }}
    .hover-meta {{ margin-top: 8px; font-size: 14px; line-height: 1.55; opacity: 0.92; }}
    .panel {{ padding: 22px; }}
    .section-label {{
      margin: 0 0 10px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      background: #edf2f7;
      color: var(--ink);
      transition: background 120ms ease, color 120ms ease, transform 120ms ease;
    }}
    .chip.is-active {{
      background: rgba(15, 118, 110, 0.16);
      color: #0b5e57;
      transform: translateY(-1px);
    }}
    .mini-card {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      margin-bottom: 12px;
    }}
    .mini-title {{ font-size: 18px; line-height: 1.35; }}
    .mini-card p {{
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    .relations-panel,
    .groups-panel {{ margin-top: 18px; padding: 22px; }}
    .relations-panel summary,
    .groups-panel summary {{
      cursor: pointer;
      font-size: 16px;
      color: var(--ink);
      padding: 6px 0;
    }}
    .rel-list {{
      margin: 14px 0 0;
      padding-left: 48px;
      max-height: 1200px;
      overflow-y: auto;
    }}
    .rel-row {{
      margin-bottom: 14px;
      padding: 10px 12px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .rel-head {{ font-size: 15px; line-height: 1.45; }}
    .rel-arrow {{
      color: var(--accent);
      font-size: 13px;
      margin: 0 4px;
    }}
    .rel-meta {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .rel-meta.rel-group {{
      text-transform: none;
      letter-spacing: 0;
      font-style: italic;
    }}
    .rel-row-expansion-only {{ display: none; }}
    body.show-expansion .rel-row-expansion-only {{ display: block; }}
    .rel-row-expansion-only.is-visible-toggle {{ display: block; }}
    .relations-toggle {{ margin: 12px 0 4px; }}
    .relations-toggle button {{
      cursor: pointer;
      padding: 8px 14px;
      font-size: 13px;
      font-family: inherit;
      color: var(--accent);
      background: rgba(15, 118, 110, 0.06);
      border: 1px solid rgba(15, 118, 110, 0.32);
      border-radius: 999px;
      transition: background 120ms ease, color 120ms ease;
    }}
    .relations-toggle button:hover,
    .relations-toggle button:focus-visible {{
      background: rgba(15, 118, 110, 0.16);
      color: #0b5e57;
      outline: none;
    }}
    .relations-toggle button[data-state="on"] {{
      background: rgba(15, 118, 110, 0.18);
      color: #0b5e57;
    }}
    .groups-stats {{
      margin: 12px 0 4px;
      padding: 12px 14px;
      background: rgba(15, 118, 110, 0.06);
      border: 1px solid rgba(15, 118, 110, 0.18);
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.65;
      color: var(--ink);
    }}
    .groups-stats strong {{ color: var(--accent); }}
    @media (max-width: 1120px) {{
      .viewer-grid {{ grid-template-columns: 1fr; }}
      .hover-card {{ position: static; min-height: 0; }}
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">BAFFLE Democritus Viewer</div>
      <h1>Relational Manifold</h1>
      <p>{esc(title)}. This viewer keeps the manifold large enough for presentation use and adds readable label context from the recovered root topics and triples. Local causal models are not rendered in this standalone viewer.</p>
    </section>
    <div class="layout">
      <section class="viewer">
        {image_markup}
        {hover_help}
      </section>
      <aside class="panel">
        <div class="section-label">Topic Labels</div>
        <div class="chip-row">{topic_markup or '<span class="chip">Root topics pending</span>'}</div>
        <div class="section-label">Key Evidence Labels</div>
        {triple_markup or '<div class="empty">Relational triples will appear here after extraction.</div>'}
        <div class="section-label">Top Local Causal Models</div>
        <div class="empty">LCM scores are produced by a separate sweep; this standalone viewer omits them.</div>
      </aside>
    </div>
    {relations_markup}
    {groups_markup}
  </div>
  <script>
    (() => {{
      const card = document.getElementById("hover-card");
      const titleNode = document.getElementById("hover-title");
      const metaNode = document.getElementById("hover-meta");
      const chips = Array.from(document.querySelectorAll("[data-topic-label]"));
      const hotspots = Array.from(document.querySelectorAll(".manifold-hotspot"));
      const clearActive = () => {{
        hotspots.forEach((node) => node.classList.remove("is-active"));
        chips.forEach((node) => node.classList.remove("is-active"));
      }};
      hotspots.forEach((node) => {{
        const activate = () => {{
          clearActive();
          node.classList.add("is-active");
          if (!card || !titleNode || !metaNode) {{
            return;
          }}
          titleNode.textContent = node.dataset.label || "Recovered cluster";
          const matchedTopic = node.dataset.topic || "";
          const sampleText = node.dataset.samples || "";
          const parts = [];
          if (matchedTopic) {{
            parts.push(`matched topic: ${{matchedTopic}}`);
          }}
          if (node.dataset.size) {{
            parts.push(`${{node.dataset.size}} manifold point(s)`);
          }}
          if (sampleText) {{
            parts.push(sampleText.split(" | ").join(" · "));
          }}
          metaNode.textContent = parts.join(" • ") || "Recovered local labels will appear here.";
          if (matchedTopic) {{
            chips.forEach((chip) => {{
              if ((chip.dataset.topicLabel || "") === matchedTopic) {{
                chip.classList.add("is-active");
              }}
            }});
          }}
        }};
        node.addEventListener("mouseenter", activate);
        node.addEventListener("focus", activate);
        node.addEventListener("click", activate);
      }});
      const stage = document.querySelector(".manifold-stage");
      if (stage) {{
        stage.addEventListener("mouseleave", () => {{
          clearActive();
          if (titleNode) {{
            titleNode.textContent = "Move over the manifold";
          }}
          if (metaNode) {{
            metaNode.textContent = "Recovered local labels will appear here.";
          }}
        }});
      }}
      // Build the "All causal relationships" rows from
      // window.RELATIONS_DATA (set by relations_data.js, loaded
      // above). Each row may carry zero, one, or two extra
      // "merged from..." meta lines for sides whose subj/obj was
      // collapsed from multiple source propositions. Uses
      // textContent throughout so payload strings cannot inject
      // markup (matches the html.escape() discipline used when this
      // section was rendered server-side).
      const relationsList = document.getElementById("relations-list");
      const relationsData =
        (window.RELATIONS_DATA && window.RELATIONS_DATA.rows) || null;
      if (relationsList && relationsData) {{
        const frag = document.createDocumentFragment();
        for (const r of relationsData) {{
          const li = document.createElement("li");
          li.className = "rel-row " + r.edge_source_class;
          const head = document.createElement("div");
          head.className = "rel-head";
          head.appendChild(document.createTextNode(r.subj));
          const arrow = document.createElement("span");
          arrow.className = "rel-arrow";
          arrow.textContent = ` — ${{r.rel_display}} → `;
          head.appendChild(arrow);
          head.appendChild(document.createTextNode(r.obj));
          li.appendChild(head);
          const meta = document.createElement("div");
          meta.className = "rel-meta";
          meta.textContent = `domain: ${{r.domain || "—"}}`;
          li.appendChild(meta);
          for (const mg of (r.member_groups || [])) {{
            const mm = document.createElement("div");
            mm.className = "rel-meta rel-group";
            mm.textContent =
              `group ${{mg.group_id}} (${{mg.side}}) — merged from: `
              + mg.members.join(" · ");
            li.appendChild(mm);
          }}
          frag.appendChild(li);
        }}
        relationsList.appendChild(frag);
      }}
      // Build the "All grouped propositions" rows from
      // window.GROUPS_DATA (set by groups_data.js, loaded above).
      // Done in JS so the sidecar can be the canonical record of
      // group membership; the HTML stays small. Uses textContent to
      // escape strings, matching the html.escape() discipline used
      // when this section was rendered server-side.
      const groupsList = document.getElementById("groups-list");
      const groupsData = (window.GROUPS_DATA && window.GROUPS_DATA.groups) || null;
      if (groupsList && groupsData) {{
        const frag = document.createDocumentFragment();
        for (const g of groupsData) {{
          const li = document.createElement("li");
          li.className = "rel-row";
          const head = document.createElement("div");
          head.className = "rel-head";
          const arrow = document.createElement("span");
          arrow.className = "rel-arrow";
          arrow.textContent = `[${{g.group_id}}]`;
          head.appendChild(arrow);
          head.appendChild(document.createTextNode(" " + g.representative));
          li.appendChild(head);
          const meta = document.createElement("div");
          meta.className = "rel-meta";
          meta.textContent =
            `members (${{g.members.length}}): ` + g.members.join(" · ");
          li.appendChild(meta);
          frag.appendChild(li);
        }}
        groupsList.appendChild(frag);
      }}
      // Toggle visibility of pure-expansion edges in the relations panel.
      document.querySelectorAll(".toggle-expansion").forEach((btn) => {{
        btn.addEventListener("click", () => {{
          const showing = document.body.classList.toggle("show-expansion");
          btn.dataset.state = showing ? "on" : "off";
          const nHidden = btn.dataset.nHidden;
          btn.textContent = showing
            ? `Hide ${{nHidden}} LLM-asserted expansion edges`
            : `Show ${{nHidden}} LLM-asserted expansion edges`;
        }});
      }});
    }})();
  </script>
</body>
</html>"""


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    run_dir: Path = (PROJECT_ROOT / RUN_DIR).resolve()
    if not run_dir.exists():
        raise SystemExit(f"FAIL: RUN_DIR does not exist: {run_dir}")

    manifold_path: Path = run_dir / "viz" / "relational_manifold_2d.png"
    if not manifold_path.exists():
        raise SystemExit(f"FAIL: manifold PNG not found: {manifold_path}")

    labels_path: Path = run_dir / "viz" / "relational_manifold_labels.json"
    if not labels_path.exists():
        raise SystemExit(f"FAIL: labels JSON not found: {labels_path}")

    triples_path: Path = run_dir / "relational_triples.jsonl"
    if not triples_path.exists():
        raise SystemExit(f"FAIL: triples JSONL not found: {triples_path}")

    root_topics_path: Path = (PROJECT_ROOT / ROOT_TOPICS_PATH).resolve()
    if not root_topics_path.exists():
        raise SystemExit(f"FAIL: ROOT_TOPICS_PATH not found: {root_topics_path}")

    output_path: Path = (PROJECT_ROOT / OUTPUT_PATH).resolve()
    if output_path.parent.resolve() != manifold_path.parent.resolve():
        raise SystemExit(
            "FAIL: OUTPUT_PATH must live in the same directory as the manifold PNG\n"
            f"  output : {output_path}\n"
            f"  png    : {manifold_path}\n"
            "Place the output at <RUN_DIR>/viz/<your_name>.html."
        )

    root_topics: List[str] = read_root_topics(root_topics_path)
    top_triples: List[Dict[str, Any]] = read_top_triples(triples_path, TOP_TRIPLE_COUNT)
    all_triples: List[Dict[str, Any]] = read_all_triples(triples_path)
    hover_clusters: List[Dict[str, Any]] = read_hover_clusters(labels_path)
    all_groups: List[Dict[str, Any]] = extract_groups(all_triples)
    compression_stats: Dict[str, Any] = compute_compression_stats(all_triples, all_groups)

    title: str = TITLE or run_dir.name
    html_str: str = build_html(
        title=title,
        manifold_filename=manifold_path.name,
        root_topics=root_topics,
        top_triples=top_triples,
        all_triples=all_triples,
        hover_clusters=hover_clusters,
        all_groups=all_groups,
        compression_stats=compression_stats,
    )

    # Sidecar data for the two large list panels. The JSON file in
    # each pair is the canonical record (consumable by any tool); the
    # JS file is the same payload wrapped as a window assignment so
    # the HTML can render the panel without an HTTP server. Both
    # files in a pair are omitted when the corresponding panel would
    # be empty (matches build_html's behaviour).
    groups_json_path:    Path = output_path.parent / "groups_data.json"
    groups_js_path:      Path = output_path.parent / "groups_data.js"
    relations_json_path: Path = output_path.parent / "relations_data.json"
    relations_js_path:   Path = output_path.parent / "relations_data.js"

    if all_groups:
        groups_payload: Dict[str, Any] = {
            "compression_stats": compression_stats,
            "n_groups":          len(all_groups),
            "groups": [
                {
                    "group_id":       g["group_id"],
                    "representative": g["representative"],
                    "members":        list(g["members"]),
                }
                for g in all_groups
            ],
        }
        json_str: str = json.dumps(groups_payload, ensure_ascii=False, indent=2)
        groups_json_path.write_text(json_str, encoding="utf-8")
        groups_js_path.write_text(
            f"window.GROUPS_DATA = {json_str};\n",
            encoding="utf-8",
        )

    if all_triples:
        relations_payload: Dict[str, Any] = {
            "n_rows": len(all_triples),
            "rows": [
                {
                    "subj":              t.get("subj") or "",
                    "obj":               t.get("obj") or "",
                    "rel_display":       _display_rel(t),
                    "domain":            t.get("domain") or "",
                    "edge_source_class": _edge_source_class(t),
                    "member_groups":     _member_groups(t),
                }
                for t in all_triples
            ],
        }
        rel_json_str: str = json.dumps(
            relations_payload, ensure_ascii=False, indent=2
        )
        relations_json_path.write_text(rel_json_str, encoding="utf-8")
        relations_js_path.write_text(
            f"window.RELATIONS_DATA = {rel_json_str};\n",
            encoding="utf-8",
        )

    output_path.write_text(html_str, encoding="utf-8")

    print(f"[ok] wrote {output_path}")
    print(f"  topics       : {len(root_topics)}")
    print(f"  top triples  : {len(top_triples)} (limit {TOP_TRIPLE_COUNT})")
    print(f"  all triples  : {len(all_triples)}")
    print(f"  clusters     : {len(hover_clusters)}")
    print(f"  groups       : {len(all_groups)} (members > 1)")
    print(
        f"  compression  : |V| {compression_stats['propositions_before']} -> "
        f"|V/~| {compression_stats['classes_after']} "
        f"({compression_stats['compression_pct']:.1f}%, "
        f"avg multi-member group {compression_stats['avg_compressed_group_size']:.2f})"
    )
    if all_groups:
        print(f"  groups   json: {groups_json_path}")
        print(f"  groups   js  : {groups_js_path}")
    if all_triples:
        print(f"  relations json: {relations_json_path}")
        print(f"  relations js  : {relations_js_path}")
    print("\nOpen in a browser:")
    print(f"  file:///{output_path.as_posix()}")


if __name__ == "__main__":
    main()

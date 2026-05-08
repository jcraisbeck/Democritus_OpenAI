#!/usr/bin/env python3
"""
quotient_report.py
------------------

Post-run comparison utility for the indiscernibility quotient.

Reads two run directories (a baseline run and a treatment run with
--quotient enabled) and writes a Markdown report with the §7.1 metrics
from final_project_written/project_description.md plus a list of the
largest equivalence classes with their member strings.

No CLI. Inputs are read from configs/quotient_config.py:
    report_baseline_dir   — path to the baseline run directory
    report_treatment_dir  — path to the treatment run directory
    report_output_path    — path to write the Markdown report

Side effects (main):
    - Reads {baseline}/relational_triples.jsonl,
            {baseline}/relational_triples.raw.jsonl (optional),
            {treatment}/relational_triples.jsonl,
            {treatment}/relational_triples.raw.jsonl.
    - Writes the Markdown report at report_output_path.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from configs.quotient_config import CONFIG


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            out.append(json.loads(raw))
    return out


def _proposition_set(triples: List[Dict[str, Any]]) -> set:
    """Set of normalized proposition strings appearing as subj or obj."""
    out: set = set()
    for t in triples:
        out.add(t["subj"].strip().lower())
        out.add(t["obj"].strip().lower())
    return out


def _count_sources(triples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count edges by `edge_source` field. Baseline triples have no
    edge_source field — treat them all as 'extraction'."""
    counts: Counter = Counter()
    for t in triples:
        source = t.get("edge_source", ["extraction"])
        if isinstance(source, str):
            source = [source]
        for s in source:
            counts[s] += 1
    return dict(counts)


def _class_size_distribution(triples: List[Dict[str, Any]]) -> Counter:
    """Class-size distribution from members_subj / members_obj fields.
    Only meaningful for treatment runs."""
    sizes: Counter = Counter()
    seen_groups: set = set()
    for t in triples:
        for side in ("subj", "obj"):
            gid = t.get(f"group_id_{side}")
            members = t.get(f"members_{side}", [])
            if gid is None or gid in seen_groups:
                continue
            seen_groups.add(gid)
            sizes[len(members)] += 1
    return sizes


def _largest_classes(
    triples: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-N largest equivalence classes by member count."""
    seen: Dict[str, List[str]] = {}
    for t in triples:
        for side in ("subj", "obj"):
            gid = t.get(f"group_id_{side}")
            members = t.get(f"members_{side}", [])
            if gid is None or len(members) < 2:
                continue
            seen[gid] = members
    ranked = sorted(seen.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [
        {"group_id": gid, "size": len(members), "members": members}
        for gid, members in ranked[:top_n]
    ]


def _format_table(rows: List[List[str]]) -> str:
    """Render a Markdown table given a list of rows (header is row 0)."""
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out_lines = []
    out_lines.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(rows[0])) + " |")
    out_lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows[1:]:
        out_lines.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out_lines)


def main() -> None:
    baseline_dir = _resolve(CONFIG["report_baseline_dir"])
    treatment_dir = _resolve(CONFIG["report_treatment_dir"])
    output_path = _resolve(CONFIG["report_output_path"])

    if not baseline_dir.is_dir():
        print(f"[report] FAIL: baseline dir not found: {baseline_dir}",
              file=sys.stderr)
        sys.exit(1)
    if not treatment_dir.is_dir():
        print(f"[report] FAIL: treatment dir not found: {treatment_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Baseline: just the standard triples file.
    baseline_triples = _read_jsonl(baseline_dir / "relational_triples.jsonl")

    # Treatment: relational_triples.jsonl is the *quotiented* file
    # after the orchestrator's rename. relational_triples.raw.jsonl
    # is the original Module 4 output preserved for provenance.
    treatment_triples = _read_jsonl(treatment_dir / "relational_triples.jsonl")
    treatment_raw = _read_jsonl(treatment_dir / "relational_triples.raw.jsonl")

    # Metrics.
    baseline_props = _proposition_set(baseline_triples)
    raw_props = _proposition_set(treatment_raw) if treatment_raw else baseline_props

    treatment_class_count = len({
        t.get("group_id_subj") for t in treatment_triples
    } | {
        t.get("group_id_obj") for t in treatment_triples
    } - {None})

    baseline_sources = _count_sources(baseline_triples)
    treatment_sources = _count_sources(treatment_triples)

    table_rows = [
        ["Metric", "Baseline", "Treatment"],
        ["#propositions (|V|)",
         str(len(baseline_props)),
         str(len(raw_props))],
        ["#extraction edges",
         str(baseline_sources.get("extraction", len(baseline_triples))),
         str(treatment_sources.get("extraction", 0))],
        ["#expansion edges",
         "0",
         str(treatment_sources.get("expansion", 0))],
        ["#equivalence classes (|V/~|)",
         str(len(baseline_props)),
         str(treatment_class_count)],
        ["#edges in G/~",
         str(len(baseline_triples)),
         str(len(treatment_triples))],
    ]

    class_sizes = _class_size_distribution(treatment_triples)
    largest = _largest_classes(treatment_triples, top_n=10)

    # Render report.
    lines: List[str] = []
    lines.append("# Indiscernibility quotient — comparison report")
    lines.append("")
    lines.append(f"**Baseline:** `{baseline_dir}`  ")
    lines.append(f"**Treatment:** `{treatment_dir}`")
    lines.append("")
    lines.append("## Quantitative")
    lines.append("")
    lines.append(_format_table(table_rows))
    lines.append("")
    lines.append("## Class-size distribution (treatment)")
    lines.append("")
    if class_sizes:
        size_rows = [["Class size", "Count"]]
        for size, count in sorted(class_sizes.items()):
            size_rows.append([str(size), str(count)])
        lines.append(_format_table(size_rows))
    else:
        lines.append("_no nontrivial classes found_")
    lines.append("")
    lines.append("## Largest equivalence classes")
    lines.append("")
    if largest:
        for entry in largest:
            lines.append(f"### {entry['group_id']} — size {entry['size']}")
            lines.append("")
            for member in entry["members"]:
                lines.append(f"- {member}")
            lines.append("")
    else:
        lines.append("_no nontrivial equivalence classes_")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    # ASCII-only output: Windows console default cp1252 chokes on Unicode arrows.
    print(f"[report] Wrote -> {output_path}")


if __name__ == "__main__":
    main()

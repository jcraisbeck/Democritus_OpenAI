"""Persistent LLM worker process for amortising per-subprocess cold start.

CLIFF's democritus pipeline spawns six LLM-only subprocess agents per query
(topic_graph x2 shards, causal_question x2 shards, causal_statement x2 shards,
relational_triple_extractor x1). Each one pays ~5-15s of Python cold start
plus ``llms.factory`` import time before running any useful code. This worker
starts once, stays warm, and dispatches subsequent jobs via a tiny
JSON-over-stdin protocol so those costs amortise to near zero for every call
after the first.

**Minimal-imports rule**: this module deliberately pre-imports ONLY the
lightweight dependencies shared by the LLM-backed scripts it dispatches
(``json``, ``runpy``, ``pathlib``, ``tqdm``, ``llms.factory``). It does NOT
import ``torch``, ``sentence_transformers``, ``umap``, ``sklearn``, or
``numpy``. Those are manifold_builder's dependencies; pulling them in here
would blow up the worker's memory footprint and defeat the purpose of
keeping a lean LLM-only worker pool. ``scripts.manifold_builder`` continues
to run as its own fresh subprocess managed directly by CLIFF.

Protocol (one JSON line per direction):

    Worker startup:   {"status": "ready"}
    Job request:      {"script": "scripts.topic_graph_builder",
                       "argv":   ["--topics-file", "...", ...],
                       "cwd":    "C:/.../run_outdir",
                       "log_path": "C:/.../agent_logs/topic_graph.log"}
    Job success:      {"status": "ok"}
    Job failure:      {"status": "error", "message": "...", "traceback": "..."}

A blank line or closed stdin triggers shutdown.

Each job runs via ``runpy.run_module(script, run_name="__main__")`` so the
target script's existing ``if __name__ == "__main__":`` argparse block
executes exactly as it would under ``python -m script --args``. Top-level
imports inside the target script hit Python's import cache after the first
run, which is where the savings come from.

Side effects: the worker ``os.chdir``s to the job's requested cwd before
running and restores the prior cwd afterwards; it redirects stdout/stderr
to the job's log file for the duration of the job so CLIFF's existing
log-collection code keeps working unchanged.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Pre-imports. Keep this list MINIMAL — only shared lightweight deps.
# Anything heavy (torch, sentence_transformers, umap, sklearn, numpy) must
# stay out of this module; add a separate worker type if a new heavy script
# needs pooling.
import tqdm  # noqa: F401  -- imported transitively by poolable scripts
from llms import factory  # noqa: F401  -- the main pre-import win


# Writing to the real stdout the parent process reads; every job temporarily
# redirects stdout to its log file, so we cache this handle at import time.
_PARENT_STDOUT = sys.stdout
_PARENT_STDERR = sys.stderr


def _write_response(response: dict) -> None:
    _PARENT_STDOUT.write(json.dumps(response) + "\n")
    _PARENT_STDOUT.flush()


def _read_job() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    stripped = line.strip()
    if not stripped:
        return None
    return json.loads(stripped)


def _run_one_job(script: str, argv: list[str], cwd: str, log_path: str) -> None:
    """Execute one job under the same semantics as ``python -m script argv``.

    Chdirs to the job's cwd, redirects stdout/stderr to the job's log file,
    sets ``sys.argv`` so argparse picks up the intended args, and dispatches
    via ``runpy.run_module`` with ``run_name="__main__"`` so the target
    script's existing entrypoint block runs unchanged. Restores the prior
    cwd and sys.argv in a ``finally`` so the worker process stays clean
    for the next job.
    """
    saved_cwd = os.getcwd()
    saved_argv = sys.argv
    os.chdir(cwd)
    try:
        sys.argv = [script] + list(argv)
        with open(log_path, "a", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                runpy.run_module(script, run_name="__main__")
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def main() -> None:
    _write_response({"status": "ready"})
    while True:
        try:
            job = _read_job()
        except json.JSONDecodeError as exc:
            _write_response({"status": "error", "message": f"invalid job JSON: {exc}"})
            continue
        if job is None:
            return
        try:
            _run_one_job(
                script=job["script"],
                argv=job["argv"],
                cwd=job["cwd"],
                log_path=job["log_path"],
            )
        except SystemExit as exc:
            # argparse errors raise SystemExit; treat nonzero as failure so
            # the pool client can surface it as a subprocess-equivalent error.
            code = exc.code if isinstance(exc.code, int) else 1
            if code == 0:
                _write_response({"status": "ok"})
            else:
                _write_response(
                    {"status": "error", "message": f"script exited with status {code}"}
                )
        except Exception:
            _write_response(
                {
                    "status": "error",
                    "message": "worker job raised",
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            _write_response({"status": "ok"})


if __name__ == "__main__":
    main()

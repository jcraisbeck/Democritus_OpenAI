#!/usr/bin/env python3
"""
run_smoke_test.py
-----------------

End-to-end smoke test for Democritus_OpenAI. Exercises Modules 0-5 and
the visualization stage on a single short PDF, into a fresh output
directory.

Usage
-----
Edit the constants in the CONFIG block below (ENV_FILE, PDF_PATH,
OUTDIR), then:

    python run_smoke_test.py

LLM credentials and provider settings (``DEMOC_LLM_PROVIDER``,
``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``, model, rate-limit, etc.)
are read from the file at ENV_FILE and injected into the subprocess
environment. The .env file is the single source of truth; this script
never copies the key elsewhere.

A PASS line is printed at the end if every expected output exists and
is non-empty. Any failed step exits with a non-zero status and a clear
error message.

The script writes two things to disk: ``configs/root_topics.txt``
(Module 0's output) and the pipeline artifacts inside OUTDIR. It does
not touch your Python environment in any other way — no venv, no
pip install. If a Democritus dependency is missing, the subprocess
call fails with ImportError and you install it yourself.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT: Path = Path(__file__).resolve().parent


# =====================================================================
# CONFIG — edit these values, then run ``python run_smoke_test.py``.
# =====================================================================

# Path to a .env file with LLM provider settings. Each non-blank,
# non-comment line is parsed as ``KEY=VALUE`` and injected into the
# subprocess environment. Interpreted relative to this script's
# directory if not absolute.
ENV_FILE: str = "../FunctorFlow/.env"

# Path to the input PDF (any short PDF; 2-3 pages is plenty for a smoke
# test). Interpreted relative to this script's directory if not absolute.
PDF_PATH: str = "../brand_democritus_block_denoise/The mortality of companies.pdf"

# Pipeline output directory, relative to this script's directory.
# Created if missing. Each run overwrites artifacts already inside it.
OUTDIR: str = "runs/smoke_test"


def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess in PROJECT_ROOT; raise CalledProcessError on non-zero exit.

    Side effects: prints the command before invoking it; streams the
    child's stdout/stderr to this process's stdout/stderr.
    """
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def discover_topics(pdf_path: Path, env: Dict[str, str]) -> None:
    """Module 0: PDF -> ``configs/root_topics.txt``. Uses the README's
    minimal-cost flags (6 root topics, 3 topics per chunk, batch 4)."""
    (PROJECT_ROOT / "configs").mkdir(exist_ok=True)
    run([
        sys.executable, "-m", "scripts.document_topic_discovery",
        "--pdf-file", str(pdf_path),
        "--num-root-topics", "6",
        "--topics-per-chunk", "3",
        "--batch-size", "4",
        "--out", "configs/root_topics.txt",
    ], env=env)


def run_pipeline(outdir: Path, env: Dict[str, str]) -> None:
    """Modules 1-5 + visualization, writing artifacts into ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "pipelines.pipeline_llm",
        "--outdir", str(outdir),
        "--domain-name", "smoke test",
    ], env=env)


def verify(outdir: Path) -> None:
    """Check that smoke-test artifacts exist and are non-empty.

    Raises SystemExit on the first failure. The chosen artifacts are
    minimal indicators that the pipeline ran end-to-end:
      - relational_triples.jsonl: triples were extracted (Module 4)
      - viz/relational_manifold_2d.png: manifold + visualization ran (Module 5)
    """
    expected: List[Path] = [
        outdir / "relational_triples.jsonl",
        outdir / "viz" / "relational_manifold_2d.png",
    ]
    print("\n[verify] checking outputs")
    for path in expected:
        if not path.exists():
            raise SystemExit(f"FAIL: missing {path}")
        size = path.stat().st_size
        if size == 0:
            raise SystemExit(f"FAIL: empty {path}")
        print(f"  ok  {path}  ({size} bytes)")


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a ``KEY -> VALUE`` dict.

    Recognized line forms:
      - ``KEY=VALUE`` (value is taken verbatim, no shell expansion)
      - blank line (skipped)
      - line whose first non-whitespace character is ``#`` (skipped)

    Surrounding double or single quotes around the value are stripped.
    Any other line shape raises ValueError so malformed files do not
    silently produce a partial environment.
    """
    out: Dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        out[key] = value
    return out


def main() -> None:
    env_file: Path = Path(ENV_FILE)
    if not env_file.is_absolute():
        env_file = PROJECT_ROOT / env_file
    env_file = env_file.resolve()
    if not env_file.exists():
        raise SystemExit(f"FAIL: ENV_FILE does not exist: {env_file}")

    env_overrides: Dict[str, str] = parse_env_file(env_file)
    if not env_overrides:
        raise SystemExit(f"FAIL: ENV_FILE has no KEY=VALUE entries: {env_file}")

    pdf_path: Path = Path(PDF_PATH)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise SystemExit(f"FAIL: PDF_PATH does not exist: {pdf_path}")

    # Subprocess environment: inherit, then overlay everything from the
    # .env file. The pipeline's LLM client reads the relevant variables
    # (DEMOC_LLM_PROVIDER, OPENAI_API_KEY / ANTHROPIC_API_KEY, model,
    # rate-limit, etc.) from os.environ.
    child_env: Dict[str, str] = os.environ.copy()
    child_env.update(env_overrides)

    print(f"[env] loaded {len(env_overrides)} variable(s) from {env_file}:")
    for key in sorted(env_overrides):
        # Mask anything that looks like a credential.
        masked = "***" if "KEY" in key or "SECRET" in key or "TOKEN" in key else env_overrides[key]
        print(f"  {key} = {masked}")

    outdir: Path = (PROJECT_ROOT / OUTDIR).resolve()

    discover_topics(pdf_path, child_env)
    run_pipeline(outdir, child_env)
    verify(outdir)

    print("\nSMOKE TEST PASSED")
    print(f"  triples : {outdir / 'relational_triples.jsonl'}")
    print(f"  figure  : {outdir / 'viz' / 'relational_manifold_2d.png'}")


if __name__ == "__main__":
    main()

"""Disk-backed cache for LLM responses.

Used by scripts/indiscernibility_quotient.py to make Steps A and B
reproducible across runs and to avoid paying for a re-issued prompt
during determinism checks.

The cache is intentionally isolated from llms/ so it can be promoted
to a shared utility later without coupling to provider internals.

Layout:
    {cache_dir}/{sha256(prompt + model + temperature)}.txt

Contents are the raw LLM response string; one file per cache entry.
Cache hits are silent; cache misses are silent on the read side and
printed once per entry on the write side.

Side effects (Cache.__init__):
    - Creates `dir` if it does not exist (when not None).
Side effects (Cache.put):
    - Writes a single file under `dir` per put.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from llms.base import LLMClient


class Cache:
    """Filesystem cache keyed by sha256(prompt + model + temperature).

    Pass `dir=None` to construct a no-op cache (every get returns None;
    every put is a silent drop). This is the cleanest way to disable
    caching from a single config flag.
    """

    def __init__(
        self,
        dir: Optional[Path],
        model: str,
        temperature: float,
    ) -> None:
        self._dir: Optional[Path] = Path(dir) if dir is not None else None
        self._model: str = model
        self._temperature: float = float(temperature)
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str) -> str:
        # Bind to model and temperature so a config change invalidates
        # cache entries automatically. Prompt is the dominant
        # contributor; model/temperature are short suffixes.
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(self._model.encode("utf-8"))
        h.update(b"\x00")
        h.update(f"{self._temperature:.6f}".encode("utf-8"))
        return h.hexdigest()

    def _path(self, prompt: str) -> Optional[Path]:
        if self._dir is None:
            return None
        return self._dir / f"{self._key(prompt)}.txt"

    def get(self, prompt: str) -> Optional[str]:
        """Return cached response or None."""
        p = self._path(prompt)
        if p is None or not p.is_file():
            return None
        return p.read_text(encoding="utf-8")

    def put(self, prompt: str, response: str) -> None:
        """Persist response. No-op if caching is disabled."""
        p = self._path(prompt)
        if p is None:
            return
        p.write_text(response, encoding="utf-8")


def cached_ask(llm: LLMClient, prompt: str, cache: Cache) -> str:
    """Get-or-call wrapper. Returns the cached response if present;
    otherwise issues `llm.ask(prompt)`, caches, and returns the result.
    """
    hit = cache.get(prompt)
    if hit is not None:
        return hit
    response = llm.ask(prompt)
    cache.put(prompt, response)
    return response

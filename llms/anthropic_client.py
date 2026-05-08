"""Anthropic (Claude) backend for the Democritus LLM factory.

This module exists **solely to accommodate Anthropic models**. The default
Democritus path in ``openai_client.py`` speaks OpenAI's chat-completions API
and remains untouched. This file is opted into by setting

    DEMOC_LLM_PROVIDER=anthropic

in the environment before ``llms.factory.make_llm_client`` constructs a
client. It is kept in a separate file (rather than folded into
``openai_client.py``) so that the Anthropic-specific schema differences —
the ``/v1/messages`` endpoint, the ``x-api-key`` auth header, the required
``anthropic-version`` header, and the list-of-content-blocks response shape
— are visibly isolated from the OpenAI-compatible path.

Environment variables read at client construction:

    ANTHROPIC_API_KEY           required
    ANTHROPIC_BASE_URL          default https://api.anthropic.com
    ANTHROPIC_MODEL             default claude-sonnet-4-6
    ANTHROPIC_API_VERSION       default 2023-06-01
    ANTHROPIC_RPM_LIMIT         default 20  — per-process RPM cap (token bucket)
    ANTHROPIC_MAX_CONCURRENCY   default 1   — in-flight HTTP cap per client
    ANTHROPIC_MAX_RETRIES       default 2   — attempts on HTTP 429 before giving up
    ANTHROPIC_RETRY_BASE_DELAY  default 5.0 — seconds; fallback when no Retry-After

The RPM token bucket is the primary rate-limit defence; retries are the
safety net for the occasional edge case. Default is sized for two LLM
pool workers sharing a 50-RPM account (2 x 20 = 40 RPM < 50). Raise
ANTHROPIC_RPM_LIMIT proportionally if your Anthropic tier allows it.

    DEMOC_LLM_MAX_TOKENS    default 256    (shared with the OpenAI path)
    DEMOC_LLM_TEMPERATURE   default 0.7    (shared with the OpenAI path)
    DEMOC_LLM_BATCH_SIZE    default 4      (shared with the OpenAI path)
    DEMOC_LLM_TIMEOUT       default 120    (shared with the OpenAI path)

Side effects: ``_single_message`` issues a synchronous outbound HTTPS POST
to the configured base URL. No files are written and no global state is
mutated.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class _TokenBucket:
    """Thread-safe token bucket for per-process RPM rate limiting.

    Tokens refill continuously at ``rate_per_minute / 60`` per second, up to
    a cap of ``rate_per_minute`` (one minute's worth of burst). ``acquire``
    blocks the calling thread until a token is available and then consumes
    one. Using this proactively on the outbound path prevents 429s from
    happening in the first place, instead of relying on retries to recover
    after the fact.

    This is a **per-process** limiter — each worker in the LLM pool has its
    own bucket. Coordination across processes is achieved by sizing the
    per-process RPM budget to ``account_rpm_limit / pool_size`` so the sum
    across workers stays under the account ceiling.
    """

    def __init__(self, rate_per_minute: float) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be > 0, got {rate_per_minute}")
        self._rate_per_second = rate_per_minute / 60.0
        self._capacity = rate_per_minute
        self._tokens = rate_per_minute
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate_per_second
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Sleep outside the lock so other threads can still refill check.
                shortfall = 1.0 - self._tokens
                wait_seconds = shortfall / self._rate_per_second
            time.sleep(wait_seconds)


# Process-wide token bucket. All AnthropicChatClient instances in the same
# process share one bucket so the per-process RPM budget is enforced even
# when multiple callers construct independent clients.
_PROCESS_BUCKET: _TokenBucket | None = None
_PROCESS_BUCKET_LOCK = threading.Lock()


def _get_process_bucket(rate_per_minute: float) -> _TokenBucket:
    global _PROCESS_BUCKET
    with _PROCESS_BUCKET_LOCK:
        if _PROCESS_BUCKET is None:
            _PROCESS_BUCKET = _TokenBucket(rate_per_minute)
        return _PROCESS_BUCKET


_ANTHROPIC_SYSTEM_PROMPT = "You are a precise assistant that follows instructions exactly."


@dataclass(frozen=True)
class AnthropicChatConfig:
    """Peer of ``OpenAIChatClient``'s env defaults, targeting Anthropic's Messages API."""

    base_url: str = "https://api.anthropic.com"
    model: str = "claude-sonnet-4-6"
    api_version: str = "2023-06-01"
    max_tokens: int = 256
    temperature: float = 0.7
    max_batch_size: int = 4
    timeout: int = 120
    # Rate limiting and retry. Primary defence is the RPM token bucket
    # (``rpm_limit``); retries exist only as a safety net for the occasional
    # burst that sneaks through. Concurrency is **deliberately** decoupled
    # from ``max_batch_size`` because pipeline scripts pass aggressive batch
    # sizes (16, 32) sized for OpenAI's serial chunk-and-loop client; using
    # those as concurrent-worker counts overwhelms Anthropic's
    # concurrent-connection cap (~5 on entry-tier) and triggers 429 cascades
    # that cost more wall time than the serial path.
    rpm_limit: float = 20.0
    max_concurrency: int = 1
    max_retries: int = 2
    retry_base_delay: float = 5.0

    @classmethod
    def from_env(cls) -> "AnthropicChatConfig":
        return cls(
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_version=os.getenv("ANTHROPIC_API_VERSION", "2023-06-01"),
            max_tokens=int(os.getenv("DEMOC_LLM_MAX_TOKENS", "256")),
            temperature=float(os.getenv("DEMOC_LLM_TEMPERATURE", "0.7")),
            max_batch_size=int(os.getenv("DEMOC_LLM_BATCH_SIZE", "4")),
            timeout=int(os.getenv("DEMOC_LLM_TIMEOUT", "120")),
            rpm_limit=float(os.getenv("ANTHROPIC_RPM_LIMIT", "20")),
            max_concurrency=int(os.getenv("ANTHROPIC_MAX_CONCURRENCY", "1")),
            max_retries=int(os.getenv("ANTHROPIC_MAX_RETRIES", "2")),
            retry_base_delay=float(os.getenv("ANTHROPIC_RETRY_BASE_DELAY", "5.0")),
        )


class AnthropicChatClient:
    """Anthropic Messages client satisfying the ``LLMClient`` protocol.

    Accepts the same kwargs as ``OpenAIChatClient`` so the factory's
    ``make_llm_client(max_tokens=..., max_batch_size=...)`` call shape works
    unchanged regardless of which provider is selected.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_batch_size: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        env_config = AnthropicChatConfig.from_env()
        self.base_url = (base_url or env_config.base_url).rstrip("/")
        self.model = model or env_config.model
        self.api_version = env_config.api_version
        self.max_tokens = max_tokens if max_tokens is not None else env_config.max_tokens
        self.temperature = temperature if temperature is not None else env_config.temperature
        # max_batch_size is accepted for ``make_llm_client`` kwarg compatibility
        # with OpenAIChatClient, but is **deliberately not used** as the
        # concurrency cap — see AnthropicChatConfig for the rationale. The
        # pipeline scripts pass values sized for OpenAI's serial chunking.
        self.max_batch_size = (
            max_batch_size if max_batch_size is not None else env_config.max_batch_size
        )
        self.max_concurrency = env_config.max_concurrency
        self.max_retries = env_config.max_retries
        self.retry_base_delay = env_config.retry_base_delay
        self.timeout = timeout if timeout is not None else env_config.timeout
        # Process-wide token bucket (first-caller wins on rate). Every
        # outbound request blocks on ``_rate_bucket.acquire()`` so we stay
        # under the account RPM ceiling proactively instead of generating
        # 429s and burning quota on retries.
        self._rate_bucket = _get_process_bucket(env_config.rpm_limit)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before running Democritus "
                "with DEMOC_LLM_PROVIDER=anthropic."
            )

    # ---------------- internal helpers ----------------

    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def _single_message(self, prompt: str) -> str:
        """One call to /v1/messages. Returns concatenated text blocks.

        Retries on HTTP 429 (rate limit) with exponential backoff because
        Anthropic's concurrent-connection and RPM caps are tight enough
        that even a well-sized batch bursts through them occasionally.
        Other HTTP errors (400, 401, 500, ...) fail fast — they aren't
        transient and retrying would just burn wall time.
        """
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": _ANTHROPIC_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            # Proactive rate limit: block until a token is available before
            # issuing the request. Under the default ANTHROPIC_RPM_LIMIT=20
            # this self-paces at ~3 seconds per call, keeping us well
            # inside a 50-RPM account even with a 2-worker pool.
            self._rate_bucket.acquire()
            request = Request(
                url,
                data=body_bytes,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = json.loads(response.read().decode(charset, errors="replace"))
                    break
            except HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s with the default base.
                    # Anthropic can also send a Retry-After header; honour it
                    # if present, otherwise fall back to the computed delay.
                    header_delay = exc.headers.get("Retry-After") if exc.headers else None
                    if header_delay is not None:
                        delay = float(header_delay)
                    else:
                        delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Anthropic request failed with HTTP {exc.code} for model "
                    f"{self.model!r} (prompt chars={len(prompt)}): {detail}"
                ) from exc
            except TimeoutError as exc:
                # SSL / socket read timeouts on long-running calls. Same
                # exponential-backoff schedule as the 429 path; raise on
                # exhaustion so the caller sees a clear failure.
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Anthropic request timed out after {self.max_retries + 1} "
                    f"attempts for model {self.model!r} (prompt chars="
                    f"{len(prompt)}, per-attempt timeout={self.timeout}s)"
                ) from exc

        # Anthropic returns a list of typed content blocks; concatenate the text ones.
        content_blocks = body["content"]
        text_parts = [
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts).strip()

    # ---------------- public API ----------------

    def ask(self, prompt: str) -> str:
        return self._single_message(prompt)

    def ask_batch(self, prompts: List[str]) -> List[str]:
        """
        Batch interface for the pipeline.

        Anthropic has no native batch endpoint, so prompts are issued as
        independent HTTP calls. Unlike ``OpenAIChatClient``'s chunk-and-loop,
        this client dispatches them concurrently via a ``ThreadPoolExecutor``,
        capped at ``self.max_concurrency`` (defaults to 2). Concurrency is
        **deliberately decoupled** from ``max_batch_size``: the pipeline
        scripts pass batch sizes of 16-32 that were sized for OpenAI's
        serial chunking, but Anthropic's concurrent-connection cap is much
        tighter (~5 on entry-tier). Combined with ``_single_message``'s 429
        retry-with-backoff, this lets us get real parallelism while staying
        inside the rate limit most of the time and self-healing when we
        don't.

        Output order is preserved to match ``prompts`` — the pipeline indexes
        results positionally, so reordering would silently corrupt downstream
        stages.
        """
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            return list(executor.map(self._single_message, prompts))

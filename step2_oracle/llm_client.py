"""llm_client.py — Async LLM client for the oracle ablation lab.

Supports two backends, auto-detected from environment:

  1. **DeepSeek direct** (Bearer auth, api.deepseek.com)
     env: DEEPSEEK_API_KEY=sk-...
          DEEPSEEK_MODEL=deepseek-chat              [optional, default]
          DEEPSEEK_BASE_URL=https://api.deepseek.com/v1   [optional]

  2. **Azure AI Foundry gateway** (api-key header, custom resource)
     env: AZURE_API_KEY=...
          AZURE_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1
          AZURE_MODEL=DeepSeek-V4-Flash            [or gpt-5.4, etc]

If both are set, Azure wins. Override via constructor args.

The client is *minimal by design* — no Pydantic, no retry decorators,
no token telemetry beyond what we save in the JSONL log. The ablation
calls each (variant, sample) exactly once and caches the raw response;
re-running picks up from the cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: dict[str, Any]


class LLMError(Exception):
    pass


class AsyncLLM:
    """Single-class client that speaks both DeepSeek and Azure wire formats.

    Detection rules in `from_env`:
      AZURE_API_KEY + AZURE_BASE_URL set    → Azure (header: api-key)
      else DEEPSEEK_API_KEY set             → DeepSeek (Bearer)
      else                                   → raise
    """

    def __init__(
        self,
        *,
        backend: str,           # "azure" or "deepseek"
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        if backend not in ("azure", "deepseek"):
            raise ValueError(f"Unknown backend: {backend!r}")
        self.backend = backend
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_env(cls, *, backend: str | None = None,
                 model: str | None = None) -> "AsyncLLM":
        # Explicit override wins.
        if backend == "azure" or (backend is None
                                  and os.environ.get("AZURE_API_KEY")
                                  and os.environ.get("AZURE_BASE_URL")):
            return cls(
                backend="azure",
                api_key=os.environ["AZURE_API_KEY"],
                base_url=os.environ["AZURE_BASE_URL"],
                model=model or os.environ.get("AZURE_MODEL", "DeepSeek-V4-Flash"),
            )
        if backend == "deepseek" or os.environ.get("DEEPSEEK_API_KEY"):
            return cls(
                backend="deepseek",
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ.get("DEEPSEEK_BASE_URL",
                                        "https://api.deepseek.com/v1"),
                model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            )
        raise LLMError(
            "No LLM credentials found. Set DEEPSEEK_API_KEY, or both "
            "AZURE_API_KEY and AZURE_BASE_URL."
        )

    async def __aenter__(self) -> "AsyncLLM":
        # HTTP/1.1 — httpx HTTP/2 needs the `h2` extra which we deliberately
        # avoided across the reboot for portability.
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *a) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        if self.backend == "azure":
            return {"api-key": self.api_key,
                    "Content-Type": "application/json"}
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def _is_reasoning_family(self) -> bool:
        """Models in the OpenAI 'reasoning' family (GPT-5, o1, gpt-5-codex)
        on Azure require `max_completion_tokens` instead of `max_tokens` and
        reject `temperature != 1`. We detect by model-string substring so
        the rule is data-driven (no hard-coded family table)."""
        m = (self.model or "").lower()
        return (m.startswith("gpt-5") or m.startswith("o1")
                or "gpt5" in m or m.startswith("o3"))

    async def chat(
        self, *, system: str, user: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        if self._client is None:
            raise LLMError("Use `async with AsyncLLM(...) as llm:`.")
        reasoning = self._is_reasoning_family()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if reasoning:
            # GPT-5 / o1 family: max_completion_tokens, temperature ignored
            # (forced to 1 server-side). We omit temperature to avoid 400s.
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        # 3 attempts with exponential backoff on transient errors.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post(
                    url, headers=self._auth_headers(), json=payload,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt)
                    continue
                # On 4xx surface the gateway's error body — without this
                # we get useless "Client error '400 Bad Request'" lines.
                if 400 <= resp.status_code < 500:
                    body_snippet = (resp.text or "")[:600].replace("\n", " ")
                    raise LLMError(
                        f"HTTP {resp.status_code} for model={self.model!r}: "
                        f"{body_snippet}"
                    )
                resp.raise_for_status()
                data = resp.json()
                latency = (time.monotonic() - t0) * 1000.0
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError(f"Empty choices: {data}")
                msg = choices[0].get("message") or {}
                text = msg.get("content") or ""
                usage = data.get("usage") or {}
                return LLMResponse(
                    text=text,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_ms=latency,
                    raw=data,
                )
            except (httpx.HTTPError, json.JSONDecodeError, LLMError) as e:
                last_exc = e
                # Don't burn retries on a 4xx — it'll fail the same way.
                if isinstance(e, LLMError) and "HTTP 4" in str(e):
                    break
                await asyncio.sleep(2 ** attempt)
        raise LLMError(f"All retries exhausted: {last_exc}")


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a single JSON object from LLM output.

    Handles three common formats:
      1. ```json {...} ```   (markdown fence)
      2. plain JSON object
      3. JSON embedded after preamble text
    """
    if not text:
        return None
    s = text.strip()
    # Strip markdown fences.
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif s.startswith("```"):
        s = s.split("```", 1)[1]
        if "```" in s:
            s = s.split("```", 1)[0]
    # Find first '{', match braces to find balanced closing.
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(s[start:end])
    except json.JSONDecodeError:
        return None

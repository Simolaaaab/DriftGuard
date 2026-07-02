"""smoke_provider.py — One-shot LLM ping to verify a provider responds.

Sends a trivial prompt to the named provider and prints the response.
Useful before launching a multi-LLM bench to confirm the model string
is recognised by the gateway.

Usage:
    python3 reboot/step5_smallbench/smoke_provider.py gpt_5_4
    python3 reboot/step5_smallbench/smoke_provider.py gpt_5_1
    python3 reboot/step5_smallbench/smoke_provider.py deepseek_v4
    python3 reboot/step5_smallbench/smoke_provider.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from llm_providers import PROVIDERS, available_providers, get_llm_client


SYSTEM = "You are a JSON-only responder. Reply with a single JSON object."
USER = (
    'Classify the URL "paypal-login-secure.tk" as either "phish" or '
    '"benign". Reply ONLY with: '
    '{"label": "phish|benign", "confidence": <float 0..1>, '
    '"reasoning": "<one sentence>"}'
)


async def ping(provider: str, *, timeout: float = 60.0) -> dict:
    spec = PROVIDERS[provider]
    print(f"\n=== {provider} (model={spec.model}, family={spec.family}) ===")
    try:
        llm_ctx = get_llm_client(provider)
    except Exception as e:
        return {"provider": provider, "ok": False,
                "error": f"client construction failed: {e}"}
    async with llm_ctx as llm:
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                llm.chat(system=SYSTEM, user=USER,
                         temperature=0.1, max_tokens=300),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"provider": provider, "ok": False,
                    "error": f"timeout after {timeout}s"}
        except Exception as e:
            return {"provider": provider, "ok": False,
                    "error": f"chat() failed: {type(e).__name__}: {e}"}
        dt = (time.monotonic() - t0) * 1000.0
        print(f"  latency: {dt:.0f} ms  "
              f"prompt_tok={resp.prompt_tokens}  "
              f"completion_tok={resp.completion_tokens}")
        print(f"  raw response (first 400 chars):")
        snippet = resp.text[:400].replace("\n", " ")
        print(f"    {snippet}")
        return {"provider": provider, "ok": True,
                "latency_ms": dt,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "raw": resp.text}


async def main_async(args) -> None:
    if args.all:
        names = available_providers()
        if not names:
            print("No providers with credentials. "
                  "Run `python3 reboot/step5_smallbench/llm_providers.py`.")
            return
    else:
        names = [args.provider]
    results = []
    for n in names:
        results.append(await ping(n, timeout=args.timeout))

    print("\n=== Summary ===")
    for r in results:
        flag = "✓" if r["ok"] else "✗"
        if r["ok"]:
            print(f"  {flag} {r['provider']:<16s} "
                  f"latency={r['latency_ms']:.0f}ms "
                  f"tokens={r['prompt_tokens']}/{r['completion_tokens']}")
        else:
            print(f"  {flag} {r['provider']:<16s} FAIL: {r['error']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("provider", nargs="?", default=None,
                   help="provider name from llm_providers.PROVIDERS "
                        "(omit when using --all)")
    p.add_argument("--all", action="store_true",
                   help="ping all providers with credentials present")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()
    if not args.all and not args.provider:
        p.error("provide a provider name or pass --all")
    if args.provider and args.provider not in PROVIDERS:
        p.error(f"unknown provider {args.provider!r}; "
                f"available: {list(PROVIDERS)}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

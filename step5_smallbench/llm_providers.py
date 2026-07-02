"""llm_providers.py — Multi-provider AsyncLLM extension.

The Step-2 `AsyncLLM` auto-detects DeepSeek-direct + Azure-AI-Foundry.
For the small-benchmark multi-LLM bench we need a thin extension that
dispatches by symbolic provider name to one of:

    deepseek_v4        AZURE_API_KEY     (Azure AI Foundry,  DeepSeek-V4-Flash)
    gpt_5_4            AZURE_OPENAI_KEY  (Azure OpenAI,      gpt-5.4)
    gpt_5_4_mini       AZURE_OPENAI_KEY  (Azure OpenAI,      gpt-5.4-mini)
    gpt_5_1            OPENAI_API_KEY    (OpenAI direct,     gpt-5.1)
    mistral_large      MISTRAL_API_KEY   (api.mistral.ai)
    grok_4             XAI_API_KEY       (api.x.ai)
    kimi_k2_6          AZURE_API_KEY     (Azure AI Foundry,  Kimi-K2.6)

All endpoints speak the OpenAI Chat Completions wire format; only
auth headers, base URLs, and model strings differ. We DO NOT modify
the existing Step-2 client — we wrap it.

Configuration is env-driven. Each entry of `PROVIDERS` declares which
env vars it needs; `available_providers()` returns the subset that has
credentials in the current shell.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step2_oracle"))
from llm_client import AsyncLLM, LLMError  # noqa: E402

log = logging.getLogger("llm_providers")


@dataclass(frozen=True)
class ProviderSpec:
    name: str                 # symbolic name used in CLI flags
    backend: str              # "azure" or "deepseek" (Bearer-style)
    env_key: str              # env var holding the API key
    env_url: str | None       # env var holding the base URL (None → default)
    default_url: str | None   # fallback URL if env_url is None or unset
    model: str                # exact model string
    family: str               # for telemetry: "deepseek" | "openai_azure" |
                              #                "openai" | "mistral" | "xai" | "kimi"


# Registry. Add entries here as new providers come online.
# Registry. Add entries here as new providers come online.
PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek_v4": ProviderSpec(
        name="deepseek_v4", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model="DeepSeek-V4-Flash",
        family="deepseek",
    ),
    "kimi_k2_6": ProviderSpec(
        name="kimi_k2_6", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model="Kimi-K2.6",
        family="kimi",
    ),
    "gpt_5_4": ProviderSpec(
        name="gpt_5_4", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model="gpt-5.4",
        family="openai_azure",
    ),
    "gpt_5_4_mini": ProviderSpec(
        name="gpt_5_4_mini", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model=os.environ.get("AZURE_GPT_5_4_MINI_MODEL", "gpt-5.4-mini"),
        family="openai_azure",
    ),
    "gpt_5_1": ProviderSpec(
        name="gpt_5_1", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model="gpt-5.1",
        family="openai",
    ),
    "Llama": ProviderSpec(
        name="Llama", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model="Llama-3.3-70B-Instruct",
        family="Llama",
    ),
    "mistral_large": ProviderSpec(
        name="mistral_large", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model=os.environ.get("MISTRAL_MODEL", "Mistral-Large-3"),
        family="mistral",
    ),
    "grok_4": ProviderSpec(
        name="grok_4", backend="azure",
        env_key="AZURE_API_KEY",
        env_url="AZURE_BASE_URL",
        default_url=None,
        model=os.environ.get("XAI_MODEL", "grok-4.3"),
        family="xai",
    ),
}


def available_providers() -> list[str]:
    """Return the names of providers whose env credentials are present."""
    out = []
    for name, spec in PROVIDERS.items():
        if not os.environ.get(spec.env_key):
            continue
        url = (os.environ.get(spec.env_url) if spec.env_url
               else spec.default_url)
        if not url:
            continue
        out.append(name)
    return out


def get_llm_client(provider_name: str) -> AsyncLLM:
    """Construct an AsyncLLM for the named provider. Raises if creds missing."""
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider_name!r}; "
            f"available registry: {list(PROVIDERS)}"
        )
    spec = PROVIDERS[provider_name]
    api_key = os.environ.get(spec.env_key)
    if not api_key:
        raise LLMError(
            f"provider {provider_name!r} requires env var "
            f"{spec.env_key} (not set)"
        )
    base_url = (os.environ.get(spec.env_url) if spec.env_url
                else spec.default_url)
    if not base_url:
        raise LLMError(
            f"provider {provider_name!r} requires env var {spec.env_url} "
            f"or a default URL"
        )
    log.info(f"[llm] provider={provider_name} backend={spec.backend} "
             f"model={spec.model} url={base_url}")
    return AsyncLLM(
        backend=spec.backend,
        api_key=api_key,
        base_url=base_url,
        model=spec.model,
    )


def provider_family(provider_name: str) -> str:
    return PROVIDERS[provider_name].family


def main_diag() -> None:
    """Diagnostic CLI: print credential availability per provider."""
    import argparse
    p = argparse.ArgumentParser(description="Diagnose configured LLM providers.")
    p.parse_args()
    print("Provider registry status:")
    print("-" * 60)
    for name, spec in PROVIDERS.items():
        key_ok = bool(os.environ.get(spec.env_key))
        url_val = (os.environ.get(spec.env_url) if spec.env_url
                   else spec.default_url)
        url_ok = bool(url_val)
        status = "✓" if (key_ok and url_ok) else "✗"
        print(f"  {status} {name:<18s}  family={spec.family:<14s} "
              f"key={spec.env_key:<22s} "
              f"{'set' if key_ok else 'MISSING':<8s}  "
              f"url={'set' if url_ok else 'MISSING'}")
    print("-" * 60)
    print(f"Available: {available_providers()}")


if __name__ == "__main__":
    main_diag()

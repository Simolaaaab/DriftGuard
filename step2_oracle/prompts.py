"""prompts.py — Five oracle prompt variants for Step-2 ablation.

The variants are designed as a 2×2 + 1 ablation:

                          probs        no probs
                       ┌─────────────┬─────────────┐
        no OSINT       │  HTML+P     │  HTML only  │  ← 'naive' baseline
                       │  (A)        │  (C)        │
                       ├─────────────┼─────────────┤
        OSINT dossier  │  HTML+P+D   │  HTML+D     │
                       │  (B)        │  (D)        │
                       └─────────────┴─────────────┘

  Plus  E: `poc_2k`  — exact PoC v1 reproduction with the original
                       2000-char HTML truncation and same prompt
                       structure. The historical baseline that scored
                       85.6% benign precision and 94.3% phish precision
                       on the PP wide batch.

The four 8K-window variants share an identical system prompt — only
the user-message body differs. This isolates "context content"
contributions from "prompt engineering" contributions.

Every variant uses the same JSON schema and the same Conservatism-Bug
fix ("decide and commit"). Comparing them is therefore a clean A/B/C/D
test of input context.

Each variant returns a tuple (system_prompt: str, user_prompt: str).
"""

from __future__ import annotations

from typing import Any, Mapping

# ── Shared system prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a cybersecurity expert analyzing a single webpage for phishing.

Reason step by step:
  1. URL — does the domain look suspicious (typosquatting, IP address,
     unusual TLD, free hosting, lookalike of a known brand)?
  2. HTML content — what does the page actually contain (password
     fields, login forms, credential-harvest JS, hidden iframes, brand
     impersonation, fake CAPTCHA, redirects, obfuscated content, fake
     wallet drainers, fake browser updates, fake stores)?
  3. Context — use any additional model opinions, OSINT signals, or
     other hints provided. These are advisory, NOT ground truth.
  4. Decide and commit. If the evidence points to phishing, label
     "phish". If the page is legitimate, label "benign". Reserve
     "uncertain" ONLY when evidence is genuinely contradictory. A
     reasoning that lists phishing patterns and then labels
     "uncertain" is a contradiction — commit to "phish".

For every indicator you find, name it explicitly. For every indicator
you actively checked but did NOT find, name that too — the downstream
verifier needs both lists.

Output a single JSON object, no prose outside the JSON, no markdown
fences:

  {"label": "phish" | "benign" | "uncertain",
   "confidence": <float in [0, 1]>,
   "reasoning": "<two to five sentences citing concrete evidence>",
   "phishing_type": "credential_harvesting" | "brand_impersonation" |
                    "scam" | "wallet_drainer" | "fake_update" |
                    "legitimate" | "other",
   "indicators_found": ["<specific indicator>", ...],
   "indicators_not_found": ["<specific check that came up empty>", ...],
   "top_factors": ["<concept the verdict depends on>", ...]}

Rules:
  - `indicators_found` MUST cite things actually present in the URL,
    HTML, or provided context.
  - `indicators_not_found` MUST be things you actively looked for.
  - Be concrete: "password field action=https://paypa1-secure.shop"
    beats "credential harvest".
  - `confidence` is your subjective probability the label is correct.
    Do not inflate.\
"""


# ── PoC v1 system prompt (faithful replication, no commit clause) ─

POC_SYSTEM_PROMPT = """\
You are a cybersecurity expert analyzing a webpage for phishing.

Analyze step by step:
  1. URL structure — suspicious?
  2. HTML content — phishing indicators?
  3. Domain-content consistency?
  4. Model opinions — when they disagree, why?

For EACH indicator: "FOUND: ..." or "NOT FOUND: what you checked".

Final verdict as JSON only:

  {"label": 0 | 1, "confidence": 0.0..1.0, "reasoning": "...",
   "phishing_type": "credential_harvesting|brand_impersonation|scam|legitimate|other",
   "indicators_found": [...], "indicators_not_found": [...]}\
"""


# ── Dossier formatter ────────────────────────────────────────────


def _opt(val: Any, default: str = "unknown") -> str:
    """Pretty-print a possibly-sentinel value."""
    if val is None or val == "" or val == -1:
        return default
    return str(val)


def format_osint_dossier(row: Mapping[str, Any]) -> str:
    """Translate the OSINT columns of a Z-matrix row into a
    discursive intelligence dossier the LLM can reason over.

    Includes BOTH the gate-dropped features (Tranco, Wayback, DNS,
    ASN) — which were unsafe for the meta-learner due to temporal
    leakage — AND the gate-kept features (RDAP, CT analytics), so
    the LLM gets the complete picture.

    The translation is *factual* (numbers in human-readable form),
    not interpretive: the LLM must reason about the implications, we
    don't pre-judge for it.
    """
    parts: list[str] = ["**Intelligence Dossier** "
                        "(passive OSINT, Tier-A sources — zero leakage):"]

    # ── Domain popularity (Tranco) ──
    in_tranco = int(row.get("is_in_tranco_1m", 0) or 0)
    rank = int(row.get("tranco_rank", -1) or -1)
    if in_tranco and rank > 0:
        if rank < 1_000:
            popnote = "extremely popular site (Top-1k)"
        elif rank < 100_000:
            popnote = "mainstream site"
        else:
            popnote = "long-tail of Top-1M"
        parts.append(f"  • Tranco rank: #{rank:,} — {popnote}.")
    else:
        parts.append("  • Tranco rank: NOT in Top-1M "
                     "(no global traffic signal).")

    # ── Wayback archival ──
    snap_n = int(row.get("wayback_n_snapshots", -1) or -1)
    first_seen = int(row.get("wayback_first_seen_days", -1) or -1)
    last_seen = int(row.get("wayback_last_seen_days", -1) or -1)
    first_ok = int(row.get("wayback_first_status_ok", -1) or -1)
    if snap_n <= 0 and first_seen < 0:
        parts.append("  • Wayback Machine: never archived "
                     "(typical of brand-new or short-lived sites).")
    else:
        bits = [f"{snap_n} snapshots"]
        if first_seen >= 0:
            yrs = first_seen / 365.25
            bits.append(f"first seen {first_seen:,} days ago "
                        f"({yrs:.1f} years)")
        if last_seen >= 0:
            bits.append(f"last snapshot {last_seen:,} days ago")
        if first_ok == 1:
            bits.append("first capture served HTTP 2xx")
        elif first_ok == 0:
            bits.append("first capture was non-2xx")
        parts.append("  • Wayback Machine: " + "; ".join(bits) + ".")

    # ── Domain registration (RDAP) ──
    age_days = int(row.get("rdap_domain_age_days", -1) or -1)
    age_risk = int(row.get("rdap_age_risk", -1) or -1)
    has_registrar = int(row.get("rdap_has_registrar", -1) or -1)
    if age_days >= 0:
        yrs = age_days / 365.25
        risk_label = {3: "high-risk (< 30 days)",
                      2: "medium-risk (30 days–1 year)",
                      1: "low-risk (> 1 year)"}.get(age_risk, "")
        parts.append(f"  • Domain age: {age_days:,} days ({yrs:.1f} years) "
                     f"{('— ' + risk_label) if risk_label else ''}.")
    else:
        parts.append("  • Domain age: unknown via RDAP "
                     "(may be a ccTLD with restricted WHOIS).")
    if has_registrar == 1:
        parts.append("  • Registrar record present in RDAP.")
    elif has_registrar == 0:
        parts.append("  • No registrar entity in RDAP "
                     "(unusual for legitimate domains).")

    # ── Certificate transparency ──
    cert_n = int(row.get("cert_history_count", -1) or -1)
    cert_first = int(row.get("cert_first_seen_days", -1) or -1)
    cert_last = int(row.get("cert_last_seen_days", -1) or -1)
    cert_div = int(row.get("cert_issuer_diversity", -1) or -1)
    cert_free = int(row.get("cert_is_free_only", -1) or -1)
    cert_wild = int(row.get("cert_has_wildcard", -1) or -1)
    cert_med = int(row.get("cert_validity_median_days", -1) or -1)
    if cert_n >= 0:
        bits = [f"{cert_n} certificate(s) in CT logs lifetime"]
        if cert_div >= 0:
            bits.append(f"{cert_div} distinct issuer(s)")
        if cert_first >= 0:
            bits.append(f"first cert {cert_first:,} days ago")
        if cert_last >= 0:
            bits.append(f"most recent {cert_last:,} days ago")
        if cert_med > 0:
            bits.append(f"median validity {cert_med} days")
        if cert_free == 1:
            bits.append("ONLY free issuers (Let's Encrypt / ZeroSSL / Buypass)")
        elif cert_free == 0:
            bits.append("mix of free and paid issuers")
        if cert_wild == 1:
            bits.append("wildcard cert observed")
        parts.append("  • Certificate Transparency: "
                     + "; ".join(bits) + ".")
    else:
        parts.append("  • Certificate Transparency: no CT log hits "
                     "(crt.sh has incomplete coverage on fresh phishing).")

    # ── DNS / ASN ──
    dns_ok = int(row.get("dns_resolved", -1) or -1)
    asn = int(row.get("asn_number", -1) or -1)
    asn_age = int(row.get("asn_age_days", -1) or -1)
    raw_country = row.get("asn_country")
    # pandas NaN → str gives 'nan'; treat that and empties as unknown.
    asn_country = (
        "" if raw_country is None or str(raw_country).lower() in ("nan", "")
        else str(raw_country).strip().upper()
    )
    if dns_ok == 1:
        bits = ["domain resolves today"]
        if asn > 0:
            bits.append(f"hosted on AS{asn}")
            if asn_country:
                bits.append(f"({asn_country})")
            if asn_age >= 0:
                yrs = asn_age / 365.25
                bits.append(f"ASN allocated {yrs:.0f} years ago")
        parts.append("  • Network: " + ", ".join(bits) + ".")
    elif dns_ok == 0:
        parts.append("  • Network: domain does NOT resolve today "
                     "(taken down or expired).")
    else:
        parts.append("  • Network: DNS resolution status unknown.")

    parts.append("")
    parts.append(
        "Use this dossier as context. Established domains "
        "(in Tranco, long Wayback history, multi-year RDAP age, "
        "rich CT diversity) lean benign; fresh registrations, "
        "no archival history, free-issuer-only certs, suspicious "
        "TLDs, and short-lived ASNs lean phishing. But OSINT alone "
        "is not a verdict — cross-reference with the HTML evidence."
    )
    return "\n".join(parts)


# ── User-message formatters ───────────────────────────────────────


def _model_block(row: Mapping[str, Any]) -> str:
    """Format the meta-learner kept-after-gate probabilities."""
    pg = float(row.get("prob_pg", 0.0) or 0.0)
    mn = float(row.get("min_prob", 0.0) or 0.0)
    sn_v = "PHISHING" if pg > 0.5 else "BENIGN"  # PG dominates after gate
    return (
        "## Static classifier opinions (post-gate, kept after Step 1)\n"
        f"  PhishGraph (HTML/graph features): {pg:.1%} → {sn_v}\n"
        f"  min(SN, PG) = {mn:.1%}\n"
    )


# Five variants. All return (system_prompt, user_prompt).

def build_variant_A(row: Mapping[str, Any], html: str) -> tuple[str, str]:
    """A = HTML (8K cleaned) + meta probabilities. The user's spec."""
    user = (
        f"URL: {row['url']}\n"
        f"Domain: {row['domain']}\n\n"
        + _model_block(row)
        + "\n## Cleaned HTML content\n```html\n"
        + html
        + "\n```\n\nEmit the JSON now per the schema in the system prompt."
    )
    return SYSTEM_PROMPT, user


def build_variant_B(row: Mapping[str, Any], html: str) -> tuple[str, str]:
    """B = HTML + probs + OSINT dossier (the strongest variant)."""
    user = (
        f"URL: {row['url']}\n"
        f"Domain: {row['domain']}\n\n"
        + _model_block(row)
        + "\n"
        + format_osint_dossier(row)
        + "\n\n## Cleaned HTML content\n```html\n"
        + html
        + "\n```\n\nEmit the JSON now per the schema in the system prompt."
    )
    return SYSTEM_PROMPT, user


def build_variant_C(row: Mapping[str, Any], html: str) -> tuple[str, str]:
    """C = HTML only (no probs, no OSINT). Naive LLM baseline."""
    user = (
        f"URL: {row['url']}\n"
        f"Domain: {row['domain']}\n\n"
        "## Cleaned HTML content\n```html\n"
        + html
        + "\n```\n\nEmit the JSON now per the schema in the system prompt."
    )
    return SYSTEM_PROMPT, user


def build_variant_D(row: Mapping[str, Any], html: str) -> tuple[str, str]:
    """D = HTML + OSINT, NO probs. Isolates the OSINT signal alone."""
    user = (
        f"URL: {row['url']}\n"
        f"Domain: {row['domain']}\n\n"
        + format_osint_dossier(row)
        + "\n\n## Cleaned HTML content\n```html\n"
        + html
        + "\n```\n\nEmit the JSON now per the schema in the system prompt."
    )
    return SYSTEM_PROMPT, user


def build_variant_poc2k(row: Mapping[str, Any], html_2k: str) -> tuple[str, str]:
    """E = PoC v1 faithful — first 2000 chars only, original prompt."""
    pg = float(row.get("prob_pg", 0.0) or 0.0)
    mn = float(row.get("min_prob", 0.0) or 0.0)
    sn_v = "PHISHING" if pg > 0.5 else "BENIGN"
    agree = "AGREE" if (pg > 0.5) == (mn > 0.5) else "DISAGREE"
    user = f"""## Webpage Information
- URL: {row['url']}
- Domain: {row['domain']}

## Model Opinions
- PhishGraph (HTML features): {pg:.1%} phishing → {sn_v}
- min(SN, PG): {mn:.1%}
- Models {agree}

## HTML Content (first 2000 chars)
```html
{html_2k[:2000]}
```

## Your Task
Analyze step by step per the system prompt.
"""
    return POC_SYSTEM_PROMPT, user


VARIANTS: dict[str, Any] = {
    "A_html_probs":       build_variant_A,
    "B_html_probs_osint": build_variant_B,
    "C_html_only":        build_variant_C,
    "D_html_osint":       build_variant_D,
    "E_poc_2k":           build_variant_poc2k,
}

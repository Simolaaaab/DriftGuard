"""verifier.py — Class-asymmetric soft trust scorer for AL oracle labels.

Architecture (per Step 4 Mini-Spec):

  T1: LLM self-confidence (raw, no transform)
  T2: Evidence verification — port of PoC v3 verifier
        - For LLM=phish: how many `indicators_found` are actually
          present in HTML/URL/OSINT?
        - For LLM=benign: how many `indicators_not_found` checks
          come back genuinely absent?
  T3: OSINT-LLM coherence — hardcoded heuristics that catch the case
        where the LLM's verdict contradicts what RDAP/Tranco/cert say.

The output `trust_score ∈ [0, 1]` becomes the `sample_weight` for
that label at retrain time. Labels are NEVER dropped (the user's
explicit directive).

Why asymmetric: phish-side trust leans on evidence (the LLM must
prove its claim); benign-side trust leans on OSINT contradictions
(an absent indicator is hard to verify in isolation but easy to
contradict via "this domain was registered 3 days ago, no Tranco").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

# ── Knowledge bases (port from oracle_single/knowledge.py, trimmed) ──

KNOWN_BRANDS = (
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "netflix", "instagram", "twitter", "linkedin", "github", "dropbox",
    "spotify", "adobe", "office365", "outlook", "gmail", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "americanexpress", "barclays", "santander", "bnp", "deutsche",
    "coinbase", "binance", "metamask", "trezor", "ledger", "kraken",
    "shopify", "stripe", "ebay", "etsy", "alibaba", "walmart",
    "fedex", "ups", "dhl", "usps",
)

FREE_HOSTING = (
    "pages.dev", "blogspot", "wordpress.com", "wixsite.com",
    "weebly.com", "netlify.app", "herokuapp.com", "github.io",
    "vercel.app", "firebaseapp.com", "workers.dev", "appspot.com",
)

SUSPICIOUS_TLDS = (
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".shop",
    ".click", ".online", ".site", ".icu", ".world", ".live",
)

# ── Data classes ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TrustAudit:
    """Per-label trust breakdown for round logging."""
    trust: float
    t1_confidence: float
    t2_evidence: float
    t3_osint: float
    n_indicators_verified: int
    n_indicators_unverifiable: int
    n_indicators_contradicted: int
    n_indicators_total: int


# ── Tier 2: evidence verification ────────────────────────────────


def _verify_positive_indicator(
    claim: str, html_lower: str, soup, url: str, domain: str,
    osint_row: dict,
) -> str:
    """Return 'verified' | 'not_found' | 'unverifiable' for a claim of
    form 'I found X in this page'."""
    c = claim.lower()

    if any(k in c for k in ("password", "login form", "credential",
                            "sign-in", "sign in")):
        if soup and soup.find_all("input", {"type": "password"}):
            return "verified"
        if re.search(r"<form[^>]*(login|signin|sign-in|auth)", html_lower):
            return "verified"
        return "not_found"

    if any(k in c for k in ("brand impersonat", "impersonat",
                            "lookalike", "claims to be")):
        title = ""
        if soup and soup.title and soup.title.string:
            title = soup.title.string.strip().lower()
        meta_desc = ""
        if soup:
            tag = soup.find("meta", attrs={"name": "description"})
            if tag and tag.get("content"):
                meta_desc = tag.get("content").lower()
        all_meta = f"{title} {meta_desc}"
        for brand in KNOWN_BRANDS:
            if brand in all_meta and brand not in domain.lower():
                return "verified"
        return "unverifiable"

    if any(k in c for k in ("typosquat",)):
        d = domain.replace("www.", "").split(".")[0].lower()
        for brand in KNOWN_BRANDS:
            if brand == d:
                continue
            if brand in d and brand not in (d,):
                return "verified"
            if len(brand) >= 4 and len(d) >= 4:
                same = sum(a == b for a, b in zip(brand, d))
                if same >= len(brand) * 0.7 and d != brand:
                    return "verified"
        return "unverifiable"

    if any(k in c for k in ("free hosting", "pages.dev", "blogspot",
                            "vercel", "netlify", "workers.dev")):
        if any(fh in domain.lower() for fh in FREE_HOSTING):
            return "verified"
        return "not_found"

    if any(k in c for k in ("ip address", "ip url", "ip-based")):
        if re.search(r"^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            return "verified"
        return "not_found"

    if any(k in c for k in ("suspicious tld", ".shop", ".xyz", ".click")):
        if any(domain.lower().endswith(t) for t in SUSPICIOUS_TLDS):
            return "verified"
        return "not_found"

    if any(k in c for k in ("wildcard cert", "wildcard certificate")):
        if int(osint_row.get("cert_has_wildcard", 0) or 0) == 1:
            return "verified"
        return "not_found"

    if any(k in c for k in ("obfuscat", "encoded", "base64",
                            "eval(", "atob(", "fromcharcode")):
        if re.search(r"(eval\s*\(|atob\s*\(|fromCharCode|\\x[0-9a-f]{2})",
                     html_lower):
            return "verified"
        return "not_found"

    if any(k in c for k in ("hidden iframe", "invisible iframe", "1x1")):
        if soup:
            for ifr in soup.find_all("iframe"):
                style = (ifr.get("style") or "").lower().replace(" ", "")
                w = ifr.get("width", "")
                h = ifr.get("height", "")
                if "display:none" in style or "visibility:hidden" in style:
                    return "verified"
                try:
                    if int(w) <= 20 and int(h) <= 20:
                        return "verified"
                except (ValueError, TypeError):
                    pass
        return "not_found"

    if any(k in c for k in ("redirect", "window.location",
                            "document.location")):
        if re.search(r"(window\.location|document\.location|meta\s+http-equiv\s*=\s*['\"]?refresh)",
                     html_lower):
            return "verified"
        return "not_found"

    if any(k in c for k in ("free cert", "let's encrypt", "letsencrypt",
                            "zerossl")):
        if int(osint_row.get("cert_is_free_only", -1) or 0) == 1:
            return "verified"
        return "not_found"

    if any(k in c for k in ("young domain", "recently registered",
                            "new domain")):
        age = int(osint_row.get("rdap_domain_age_days", -1) or -1)
        if 0 <= age <= 90:
            return "verified"
        if age > 365:
            return "not_found"
        return "unverifiable"

    return "unverifiable"


def _verify_absent_indicator(
    claim: str, html_lower: str, soup, url: str, domain: str,
    osint_row: dict,
) -> str:
    """Return 'verified' | 'contradicted' | 'unverifiable' for an
    absence claim of form 'I checked X and did not find it'."""
    c = claim.lower()

    if any(k in c for k in ("password", "login form", "credential")):
        if soup and soup.find_all("input", {"type": "password"}):
            return "contradicted"
        if re.search(r"<form[^>]*(login|signin|auth)", html_lower):
            return "contradicted"
        return "verified"

    if "hidden iframe" in c:
        if soup:
            for ifr in soup.find_all("iframe"):
                style = (ifr.get("style") or "").lower().replace(" ", "")
                w = ifr.get("width", "")
                h = ifr.get("height", "")
                if "display:none" in style or "visibility:hidden" in style:
                    return "contradicted"
                try:
                    if int(w) <= 20 and int(h) <= 20:
                        return "contradicted"
                except (ValueError, TypeError):
                    pass
        return "verified"

    if any(k in c for k in ("brand impersonat", "brand mismatch")):
        title = ""
        if soup and soup.title and soup.title.string:
            title = soup.title.string.lower()
        for brand in KNOWN_BRANDS:
            if brand in title and brand not in domain.lower():
                return "contradicted"
        return "verified"

    if any(k in c for k in ("typosquat",)):
        # Hard to disprove without exhaustive search → give benefit.
        return "verified"

    if any(k in c for k in ("obfuscat", "credential harvest", "eval(")):
        if re.search(r"(eval\s*\(|atob\s*\(|fromCharCode)", html_lower):
            return "contradicted"
        return "verified"

    return "unverifiable"


def evidence_trust(label: int, indicators_found, indicators_not_found,
                   html_lower: str, soup, url: str, domain: str,
                   osint_row: dict) -> tuple[float, int, int, int, int]:
    """Compute T2 evidence trust + audit counts.

    Returns (t2, n_verified, n_unverifiable, n_contradicted, n_total).
    """
    n_v = n_u = n_c = 0

    if label == 1:
        items = [str(x).strip() for x in (indicators_found or [])
                 if isinstance(x, str) and len(str(x).strip()) >= 5]
        if not items:
            return 0.5, 0, 0, 0, 0
        for claim in items:
            r = _verify_positive_indicator(claim, html_lower, soup,
                                            url, domain, osint_row)
            if r == "verified":      n_v += 1
            elif r == "unverifiable": n_u += 1
            else:                     n_c += 1     # not_found
        t2 = (n_v + 0.5 * n_u) / len(items)
        return t2, n_v, n_u, n_c, len(items)

    # benign
    items = [str(x).strip() for x in (indicators_not_found or [])
             if isinstance(x, str) and len(str(x).strip()) >= 5]
    if not items:
        return 0.5, 0, 0, 0, 0
    for claim in items:
        r = _verify_absent_indicator(claim, html_lower, soup,
                                      url, domain, osint_row)
        if r == "verified":       n_v += 1
        elif r == "unverifiable":  n_u += 1
        else:                      n_c += 1     # contradicted
    t2 = (n_v + 0.5 * n_u) / len(items)
    if n_c > 0:
        t2 *= 0.5
    return t2, n_v, n_u, n_c, len(items)


# ── Tier 3: OSINT-LLM coherence ──────────────────────────────────


def osint_coherence(label: int, osint_row: dict) -> float:
    """Hardcoded heuristics. Calibrated against the empirical AL-stream
    asymmetry: LLM=phish is ~99% precise (trust high by default),
    LLM=benign is ~59% precise (let OSINT challenge it).
    """
    in_tranco = int(osint_row.get("is_in_tranco_1m", 0) or 0)
    rank = int(osint_row.get("tranco_rank", -1) or -1)
    age = int(osint_row.get("rdap_domain_age_days", -1) or -1)
    cert_n = int(osint_row.get("cert_history_count", -1) or -1)
    cert_free = int(osint_row.get("cert_is_free_only", -1) or -1)
    wayback_n = int(osint_row.get("wayback_n_snapshots", -1) or -1)
    dns_ok = int(osint_row.get("dns_resolved", -1) or -1)
    domain = str(osint_row.get("domain") or "").lower()

    # Subdomain-of-free-hosting fix: RDAP returns the PARENT's age.
    # *.vercel.app inherits ~9yr, *.blogspot.com ~20yr, etc., even if
    # the subdomain was created yesterday for phishing. Neutralise the
    # 'is_established' criterion in these cases.
    is_subdomain_of_freehost = any(fh in domain for fh in FREE_HOSTING)
    if is_subdomain_of_freehost:
        # The Wayback signal is per-subdomain (separate from parent age),
        # so it survives this neutralisation.
        age = -1                   # treat parent age as unknown
        in_tranco = 0              # Tranco rank applies to parent too

    is_established = (
        (in_tranco == 1 and 0 < rank < 100_000)
        or (age >= 5 * 365)
        or (wayback_n > 100 and age >= 365)
    )
    is_young = (0 <= age < 30) or (dns_ok == 0)
    is_throwaway = (
        is_young
        or (cert_free == 1 and 0 <= cert_n < 5)
        or (in_tranco == 0 and 0 <= age < 90)
    )

    if label == 1:    # LLM says phish
        # The LLM commits to phish only when confident; precision ~99%
        # on the AL stream. OSINT should reinforce, not contradict
        # unless very strong evidence of established legit.
        if is_throwaway:                    return 1.0
        if is_young:                        return 0.85
        if is_established:                  return 0.15  # strong contradict
        if in_tranco == 1 and 0 < rank < 100_000: return 0.30
        if in_tranco == 1:                  return 0.55
        return 0.65                         # default lean toward LLM

    # LLM says benign — the high-noise side. OSINT contradicts vigorously.
    if is_established:                      return 1.00
    # Mild Tranco signal — needs age to confirm legit.
    if in_tranco == 1 and 0 < rank < 1_000_000:
        if age >= 1095:                     return 0.85    # 3+yr
        if age >= 365:                      return 0.65    # 1+yr
        return 0.45                          # young but in Tranco
    # Not in Tranco at all.
    if age >= 1825:                         return 0.55    # 5+yr niche
    if age >= 365:                          return 0.30    # tail-of-internet
    if is_throwaway:                        return 0.05    # ⚠ contradicted
    if is_young:                            return 0.15
    return 0.25                              # unknown, lean suspicious


# ── Public API ───────────────────────────────────────────────────


def _full_trust(label: int, t1: float, t2: float, t3: float) -> float:
    """Asymmetric combination calibrated against empirical class
    precision on the AL stream.

      label=phish (~99% precision):  trust leans on the LLM, T2/T3
                                     mostly confirmatory. Set a floor
                                     so we don't penalise the reliable
                                     class for hard-to-verify text claims.

      label=benign (~59% precision): trust leans on T3 OSINT contradiction.
                                     The LLM is confidently wrong on ~40%
                                     of benign calls (Mode-2 .shop pattern).
                                     T3 must dominate to filter those.
    """
    if label == 1:
        # 50% evidence, 30% confidence, 20% OSINT — but with a floor of 0.55
        # because the empirical phish precision is 99%.
        raw = 0.5 * t2 + 0.3 * t1 + 0.2 * t3
        return max(raw, 0.55) if t3 > 0.20 else raw
    # benign side: OSINT dominates.
    return 0.6 * t3 + 0.2 * t1 + 0.2 * t2


def compute_trust(
    *,
    label: int,
    confidence: float,
    indicators_found: list,
    indicators_not_found: list,
    raw_html: str | None,
    url: str,
    domain: str,
    osint_row: dict,
    mode: str = "full",
) -> TrustAudit:
    """Score one oracle label. `mode` selects which tiers are active:

      'off'              → trust=1.0 (no verifier)
      't1_only'          → trust = T1 (confidence)
      't2_only'          → trust = T2 (evidence verification)
      't3_only'          → trust = T3 (OSINT coherence)
      'full'             → trust = asymmetric combination of T1+T2+T3
      'hard_filter'      → like 'full' but trust → 0 if < 0.5
    """
    if mode == "off":
        return TrustAudit(
            trust=1.0, t1_confidence=confidence, t2_evidence=1.0, t3_osint=1.0,
            n_indicators_verified=0, n_indicators_unverifiable=0,
            n_indicators_contradicted=0, n_indicators_total=0,
        )

    t1 = float(confidence or 0.0)

    soup = None
    html_lower = ""
    if raw_html:
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            html_lower = raw_html.lower()
        except Exception:
            soup = None

    t2, n_v, n_u, n_c, n_total = evidence_trust(
        label, indicators_found, indicators_not_found,
        html_lower, soup, url, domain, osint_row,
    )
    t3 = osint_coherence(label, osint_row)

    if mode == "t1_only":
        trust = t1
    elif mode == "t2_only":
        trust = t2
    elif mode == "t3_only":
        trust = t3
    elif mode == "hard_filter":
        combined = _full_trust(label, t1, t2, t3)
        trust = combined if combined >= 0.5 else 0.0
    else:   # 'full'
        trust = _full_trust(label, t1, t2, t3)

    trust = max(0.0, min(1.0, trust))
    return TrustAudit(
        trust=trust, t1_confidence=t1, t2_evidence=t2, t3_osint=t3,
        n_indicators_verified=n_v, n_indicators_unverifiable=n_u,
        n_indicators_contradicted=n_c, n_indicators_total=n_total,
    )

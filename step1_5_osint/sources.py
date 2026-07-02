"""sources.py — Async low-level clients for Tier-A OSINT sources.

Each fetch returns a dict shaped like:

    {
        "source":          "<short name>",
        "fetched_at_utc":  "<iso8601>",
        "ok":              bool,
        "error":           "<type: message>"  (only when ok=False),
        ...source-specific raw payload...
    }

No caching, no rate limiting, no retries here. The orchestrator
(`collect_osint.py`) is responsible for those. Each function is
self-contained, never raises, and never logs — the caller decides.

Sources (all Tier A — zero or near-zero leakage):
  - crt.sh                  CT log issuance history
  - Wayback CDX API         archival timeline (the "dead site" lifesaver)
  - RDAP (rdap.org)         registration date + registrar
  - BGPView (ip / asn)      ASN + allocation date
  - System DNS resolve      A records + canonical hostname

Tranco rank is offline — see TrancoIndex in collect_osint.py.
"""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import datetime, timezone

import httpx


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _err(source: str, exc: BaseException, **extra) -> dict:
    return {
        "source": source,
        "fetched_at_utc": _now_utc(),
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        **extra,
    }


# ── crt.sh — Certificate Transparency ─────────────────────────────


async def fetch_crtsh(client: httpx.AsyncClient, domain: str) -> dict:
    """Certificate history for the domain.

    The query returns *all* certs that include the domain in any SAN.
    The orchestrator will deduplicate later if needed. Sorting is left
    to the feature-derivation step.
    """
    url = f"https://crt.sh/?q={domain}&output=json"
    try:
        r = await client.get(url, timeout=45.0)
        if r.status_code == 200:
            text = r.text.strip()
            if not text:
                return {
                    "source": "crtsh", "fetched_at_utc": _now_utc(),
                    "ok": True, "certs": [], "n": 0,
                }
            certs = json.loads(text)
            return {
                "source": "crtsh", "fetched_at_utc": _now_utc(),
                "ok": True, "certs": certs, "n": len(certs),
            }
        return {
            "source": "crtsh", "fetched_at_utc": _now_utc(),
            "ok": False, "error": f"http_{r.status_code}",
        }
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        return _err("crtsh", exc)


# ── Wayback Machine CDX API ───────────────────────────────────────


async def fetch_wayback(client: httpx.AsyncClient, domain: str) -> dict:
    """Wayback CDX: timeline of HTTP snapshots for the domain.

    Uses `matchType=domain` which captures any URL containing the
    domain. Returned columns: timestamp, original, statuscode, length.
    First row is the column header.

    The output is sorted chronologically. We cap at 1000 snapshots —
    more than enough to bound first-seen / last-seen / count.
    """
    url = (
        "http://web.archive.org/cdx/search/cdx"
        f"?url={domain}/&matchType=domain&output=json"
        "&fl=timestamp,original,statuscode,length&limit=1000"
    )
    try:
        r = await client.get(url, timeout=45.0)
        if r.status_code == 200:
            txt = r.text.strip()
            if not txt:
                return {
                    "source": "wayback", "fetched_at_utc": _now_utc(),
                    "ok": True, "snapshots": [], "n": 0,
                }
            data = json.loads(txt)
            # Drop the header row.
            rows = data[1:] if data else []
            return {
                "source": "wayback", "fetched_at_utc": _now_utc(),
                "ok": True, "snapshots": rows, "n": len(rows),
            }
        return {
            "source": "wayback", "fetched_at_utc": _now_utc(),
            "ok": False, "error": f"http_{r.status_code}",
        }
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        return _err("wayback", exc)


# ── RDAP — Registration data (registrar + dates) ──────────────────


def _registrable_domain(domain: str) -> str:
    """Strip subdomains. Handle .co.uk / .com.au style multi-part TLDs.

    Not a full PSL implementation — RDAP itself tolerates subdomains
    on most queries, but registrable form gives the highest hit rate.
    """
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


async def fetch_rdap(client: httpx.AsyncClient, domain: str) -> dict:
    """rdap.org universal proxy. Returns the raw RDAP JSON."""
    reg = _registrable_domain(domain)
    url = f"https://rdap.org/domain/{reg}"
    try:
        r = await client.get(url, timeout=25.0, follow_redirects=True)
        if r.status_code == 200:
            return {
                "source": "rdap", "fetched_at_utc": _now_utc(),
                "ok": True, "registrable": reg, "data": r.json(),
            }
        return {
            "source": "rdap", "fetched_at_utc": _now_utc(),
            "ok": False, "registrable": reg, "error": f"http_{r.status_code}",
        }
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        return _err("rdap", exc, registrable=reg)


# ── DNS — system resolver (offloaded to a thread) ─────────────────


async def fetch_dns(domain: str) -> dict:
    """System DNS A record resolution via getaddrinfo / gethostbyname.

    `socket.gethostbyname_ex` is blocking — wrap it in the default
    executor so the async loop stays responsive.
    """
    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(None, socket.gethostbyname_ex, domain)
        canonical, aliases, ips = info
        return {
            "source": "dns", "fetched_at_utc": _now_utc(),
            "ok": True, "canonical": canonical,
            "aliases": list(aliases or []),
            "ips": list(ips or []),
        }
    except (socket.gaierror, socket.timeout, OSError) as exc:
        return _err("dns", exc)


# ── RIPE Stat — IP → ASN, ASN → allocation metadata ──────────────
# We use RIPE Stat (https://stat.ripe.net/docs/02.data-api/) instead
# of BGPView because (a) it is the official RIPE NCC service and is
# reachable from environments where api.bgpview.io has DNS issues,
# (b) it carries the WHOIS RegDate which is the authoritative ASN
# allocation date (BGPView's date_allocated occasionally drifts),
# (c) the documented limit is 1000 req/min — very generous.


async def fetch_ripestat_network(client: httpx.AsyncClient, ip: str) -> dict:
    """RIPE Stat network-info: IP → ASN(s) and routed prefix."""
    url = f"https://stat.ripe.net/data/network-info/data.json?resource={ip}"
    try:
        r = await client.get(url, timeout=25.0)
        if r.status_code == 200:
            return {
                "source": "ripestat_network", "fetched_at_utc": _now_utc(),
                "ok": True, "ip": ip, "data": r.json(),
            }
        return {
            "source": "ripestat_network", "fetched_at_utc": _now_utc(),
            "ok": False, "ip": ip, "error": f"http_{r.status_code}",
        }
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        return _err("ripestat_network", exc, ip=ip)


async def fetch_ripestat_asn(client: httpx.AsyncClient, asn: int | str) -> dict:
    """RIPE Stat WHOIS for an ASN — carries the RegDate (allocation),
    the AS holder, and the country code where applicable.

    The whois data-call is preferred over as-overview because
    as-overview omits the registration date — we need that for the
    `asn_age_days` feature.
    """
    asn_clean = str(asn).removeprefix("AS")
    url = f"https://stat.ripe.net/data/whois/data.json?resource=AS{asn_clean}"
    try:
        r = await client.get(url, timeout=25.0)
        if r.status_code == 200:
            return {
                "source": "ripestat_asn", "fetched_at_utc": _now_utc(),
                "ok": True, "asn": int(asn_clean), "data": r.json(),
            }
        return {
            "source": "ripestat_asn", "fetched_at_utc": _now_utc(),
            "ok": False, "asn": int(asn_clean), "error": f"http_{r.status_code}",
        }
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        return _err("ripestat_asn", exc,
                    asn=int(asn_clean) if asn_clean.isdigit() else asn_clean)

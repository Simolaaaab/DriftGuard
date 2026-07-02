"""collect_osint.py — Async OSINT collection orchestrator (Tier-A only).

What this does
──────────────
Reads the per-dataset JSON files (KP/DP/PP), extracts unique domains,
fans out async queries to Tier-A OSINT sources, and persists every
single raw response to a deterministic cache file so the whole
collection is resumable, reproducible, and tagged with provenance.

What it does NOT do
───────────────────
 - No feature derivation (see `derive_features.py`).
 - No Z-matrix merging (see `enrich_z_matrices.py`).
 - No Tier-B / Tier-C sources. VirusTotal, GSB, PhishTank, AbuseIPDB
   and friends are deliberately excluded — they are likely trained on
   the same public corpora as our ground truth. Adding them would
   inflate cross-dataset numbers via label leakage. See README.md.

Sources in this step
────────────────────
 1. Tranco        offline rank lookup (downloads top-1m.csv once)
 2. crt.sh        CT log issuance history
 3. Wayback CDX   archival timeline of the domain
 4. RDAP          registration date + registrar (Tier A only when used
                  *purely as data*, not as registrar-reputation score)
 5. DNS           system resolver — for IP needed by BGPView
 6. BGPView /ip   IP → ASN
 7. BGPView /asn  ASN → allocation metadata

Cache layout
────────────
    reboot/step1_5_osint/cache/
    ├── tranco/top1m.csv                   (downloaded once)
    └── raw/
        ├── crtsh/<domain>.json.gz
        ├── wayback/<domain>.json.gz
        ├── rdap/<domain>.json.gz
        ├── dns/<domain>.json.gz
        ├── bgpview_ip/<ip>.json.gz
        └── bgpview_asn/<asn>.json.gz

Each file is the *raw* response with a `fetched_at_utc` field. Reading
the cache is O(1); the orchestrator skips fetches whose cache file
already exists. Delete a specific cache directory to force re-fetch
of that source only.

Usage
─────
    # one-time prep (if not already done)
    python reboot/step1_ensemble/prepare_kp_json.py

    # collect all three datasets in one async run
    python reboot/step1_5_osint/collect_osint.py --datasets kp,dp,pp

    # resume / refresh single dataset
    python reboot/step1_5_osint/collect_osint.py --datasets pp

    # limit for smoke testing
    python reboot/step1_5_osint/collect_osint.py --datasets pp --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import logging
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from sources import (  # noqa: E402
    fetch_crtsh,
    fetch_dns,
    fetch_rdap,
    fetch_ripestat_asn,
    fetch_ripestat_network,
    fetch_wayback,
)

# ── Paths ─────────────────────────────────────────────────────────

REBOOT = Path(__file__).resolve().parent.parent
DATASET_JSONS = {
    "kp": REBOOT / "datasets" / "knowphish_data.json",
    "dp": REBOOT / "datasets" / "deltaphish_data.json",
    "pp": REBOOT / "datasets" / "phreshphish_data.json",
}
CACHE_ROOT = THIS.parent / "cache"
RAW_DIR = CACHE_ROOT / "raw"
TRANCO_DIR = CACHE_ROOT / "tranco"
TRANCO_FILE = TRANCO_DIR / "top1m.csv"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

# ── Per-source rate limits ────────────────────────────────────────
# Semaphore concurrency + minimum per-call interval. Conservative
# defaults — these APIs are free, treat them with respect.

RATE_LIMITS = {
    "crtsh":              {"concurrency": 2, "min_interval_s": 1.0},
    "wayback":            {"concurrency": 4, "min_interval_s": 0.3},
    "rdap":               {"concurrency": 2, "min_interval_s": 0.6},
    "ripestat_network":   {"concurrency": 6, "min_interval_s": 0.15},
    "ripestat_asn":       {"concurrency": 6, "min_interval_s": 0.15},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("collect_osint")


# ── Tranco download (one-time) ────────────────────────────────────


def ensure_tranco() -> Path:
    """Download Tranco top-1m.csv if absent. Returns the local path."""
    if TRANCO_FILE.exists() and TRANCO_FILE.stat().st_size > 1_000_000:
        return TRANCO_FILE
    TRANCO_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Tranco top-1m.csv.zip ...")
    with httpx.Client(timeout=120.0, follow_redirects=True) as c:
        r = c.get(TRANCO_URL)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        # Inside the zip is a single CSV with name like "top-1m.csv".
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(member) as src, TRANCO_FILE.open("wb") as dst:
            dst.write(src.read())
    log.info(f"Tranco saved to {TRANCO_FILE} ({TRANCO_FILE.stat().st_size:,} bytes)")
    return TRANCO_FILE


# ── Cache helpers ─────────────────────────────────────────────────


def _cache_path(source: str, key: str) -> Path:
    safe = key.replace("/", "_")[:200] or "_empty"
    return RAW_DIR / source / f"{safe}.json.gz"


def _cache_load(source: str, key: str) -> dict | None:
    p = _cache_path(source, key)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_save(source: str, key: str, payload: dict) -> None:
    p = _cache_path(source, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


# ── Rate-limited fetch wrappers ───────────────────────────────────


class RateLimiter:
    """Per-source concurrency + minimum-interval limiter.

    `acquire()` blocks until both (a) a semaphore slot is free and
    (b) at least `min_interval_s` has elapsed since the last release.
    """

    def __init__(self, concurrency: int, min_interval_s: float) -> None:
        self.sem = asyncio.Semaphore(concurrency)
        self.min_interval = min_interval_s
        self._last_release = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self.sem.acquire()
        # Throttle on release timing — guarantees inter-call spacing
        # even when concurrency=1.
        async with self._lock:
            wait = self._last_release + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_release = time.monotonic()

    async def __aexit__(self, *a) -> None:
        self.sem.release()


@dataclass
class Limiters:
    crtsh: RateLimiter
    wayback: RateLimiter
    rdap: RateLimiter
    ripestat_network: RateLimiter
    ripestat_asn: RateLimiter

    @classmethod
    def from_rate_limits(cls) -> "Limiters":
        return cls(
            crtsh=RateLimiter(**RATE_LIMITS["crtsh"]),
            wayback=RateLimiter(**RATE_LIMITS["wayback"]),
            rdap=RateLimiter(**RATE_LIMITS["rdap"]),
            ripestat_network=RateLimiter(**RATE_LIMITS["ripestat_network"]),
            ripestat_asn=RateLimiter(**RATE_LIMITS["ripestat_asn"]),
        )


# Error classification for caching policy.
# Permanent: the API will give the same answer next time — cache it.
# Transient: server/network hiccup — don't cache, let resume retry.
_PERMANENT_ERROR_MARKERS = (
    "http_404", "http_410", "http_422",
    "gaierror",         # NXDOMAIN — domain really doesn't resolve
    "JSONDecodeError",  # body wasn't JSON; for RDAP this means "no record"
)
_TRANSIENT_ERROR_MARKERS = (
    "http_429", "http_500", "http_502", "http_503", "http_504",
    "ConnectError", "ReadTimeout", "WriteTimeout", "TimeoutException",
    "RemoteProtocolError", "PoolTimeout", "exhausted_retries",
)


def _is_permanent_error(payload: dict) -> bool:
    err = str(payload.get("error", ""))
    return any(m in err for m in _PERMANENT_ERROR_MARKERS)


def _is_transient_error(payload: dict) -> bool:
    err = str(payload.get("error", ""))
    return any(m in err for m in _TRANSIENT_ERROR_MARKERS)


async def _cached(
    source: str,
    key: str,
    fetcher,                    # async callable that returns dict
    limiter: RateLimiter,
) -> dict:
    """Cache-or-fetch shim with error-class-aware caching policy.

    Cache hit: short-circuit.
    Cache miss → fetch under rate limiter.
      ok=True            → cache (permanent fact).
      permanent error    → cache (won't change on retry).
      transient error    → DO NOT cache (let next resume retry).
      anything else      → cache defensively (unknown class).
    """
    hit = _cache_load(source, key)
    if hit is not None:
        return hit
    async with limiter:
        result = await fetcher()
    if result.get("ok"):
        _cache_save(source, key, result)
    elif _is_transient_error(result):
        # don't cache — let next resume retry
        pass
    elif _is_permanent_error(result):
        _cache_save(source, key, result)
    else:
        # unknown — cache defensively so we don't spin forever on
        # weird edge cases; user can `rm` to retry
        _cache_save(source, key, result)
    return result


# ── Per-domain orchestration ──────────────────────────────────────


async def collect_one_domain(
    domain: str, client: httpx.AsyncClient, limiters: Limiters,
) -> dict:
    """Run all sources for a single domain. Returns aggregated result."""
    # crt.sh, Wayback, RDAP are independent → run in parallel.
    crtsh_task = _cached("crtsh", domain,
                         lambda: fetch_crtsh(client, domain), limiters.crtsh)
    wayback_task = _cached("wayback", domain,
                           lambda: fetch_wayback(client, domain), limiters.wayback)
    rdap_task = _cached("rdap", domain,
                        lambda: fetch_rdap(client, domain), limiters.rdap)

    # DNS → RIPE Stat chain has internal dependency. Resolve DNS first.
    async def dns_chain() -> tuple[dict, dict | None, dict | None]:
        dns_hit = _cache_load("dns", domain)
        if dns_hit is None:
            dns_hit = await fetch_dns(domain)
            _cache_save("dns", domain, dns_hit)
        if not dns_hit.get("ok") or not dns_hit.get("ips"):
            return dns_hit, None, None
        ip = dns_hit["ips"][0]
        net_res = await _cached(
            "ripestat_network", ip,
            lambda: fetch_ripestat_network(client, ip), limiters.ripestat_network,
        )
        asn_res = None
        if net_res.get("ok"):
            # RIPE Stat returns asns as a list of stringified ASN numbers.
            asns = ((net_res.get("data") or {}).get("data") or {}).get("asns") or []
            asn_num = None
            for a in asns:
                try:
                    asn_num = int(str(a).removeprefix("AS"))
                    break
                except (ValueError, AttributeError):
                    continue
            if asn_num is not None:
                asn_res = await _cached(
                    "ripestat_asn", str(asn_num),
                    lambda: fetch_ripestat_asn(client, asn_num),
                    limiters.ripestat_asn,
                )
        return dns_hit, net_res, asn_res

    dns_task = dns_chain()

    crtsh, wayback, rdap, (dns, ip_info, asn_info) = await asyncio.gather(
        crtsh_task, wayback_task, rdap_task, dns_task,
    )

    return {
        "domain": domain,
        "crtsh": crtsh,
        "wayback": wayback,
        "rdap": rdap,
        "dns": dns,
        "ripestat_network": ip_info,
        "ripestat_asn": asn_info,
    }


# ── Dataset loaders ───────────────────────────────────────────────


def domains_from_json(path: Path) -> list[str]:
    """Extract the unique set of domains from a dataset JSON, preserving
    first-seen order so resumability is deterministic."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset JSON: {path}")
    with path.open() as f:
        entries = json.load(f)
    seen: set[str] = set()
    out: list[str] = []
    for e in entries:
        url = (e.get("url") or e.get("final_url") or "").strip()
        domain = (e.get("domain_name") or "").strip().lower()
        if not domain and url:
            try:
                host = urlparse(url if "://" in url else "http://" + url).netloc
                domain = host.lower().split(":")[0]
            except Exception:
                domain = ""
        if domain and domain not in seen:
            seen.add(domain)
            out.append(domain)
    return out


# ── Main loop ─────────────────────────────────────────────────────


async def collect_dataset(tag: str, domains: list[str], limit: int = 0) -> None:
    if limit:
        domains = domains[:limit]
    log.info(f"[{tag}] starting collection for {len(domains)} domains")

    limiters = Limiters.from_rate_limits()
    headers = {"User-Agent": "PhishHook-Reboot-Research/1.0"}
    # HTTP/1.1 is fine for our free-API rates. HTTP/2 would require
    # the 'h2' extra; not worth a new dep for the marginal gain.
    transport = httpx.AsyncHTTPTransport(retries=1)
    async with httpx.AsyncClient(
        headers=headers, transport=transport,
    ) as client:
        t0 = time.monotonic()
        for i, d in enumerate(domains, 1):
            try:
                await collect_one_domain(d, client, limiters)
            except Exception as exc:
                log.warning(f"[{tag}] {d} unexpected exception: {exc!r}")
            if i % 50 == 0 or i == len(domains):
                el = time.monotonic() - t0
                eta = el * (len(domains) - i) / i if i else 0
                log.info(
                    f"[{tag}] {i}/{len(domains)}  "
                    f"elapsed={el/60:.1f}m  eta={eta/60:.1f}m"
                )
    log.info(f"[{tag}] done in {(time.monotonic() - t0)/60:.1f} minutes")


async def main_async(args) -> None:
    ensure_tranco()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    requested = [t.strip() for t in args.datasets.split(",") if t.strip()]
    for tag in requested:
        if tag not in DATASET_JSONS:
            log.error(f"Unknown dataset tag {tag!r}; expected kp/dp/pp")
            continue
        domains = domains_from_json(DATASET_JSONS[tag])
        log.info(f"[{tag}] dataset JSON: {DATASET_JSONS[tag]}  unique domains: {len(domains)}")
        await collect_dataset(tag, domains, limit=args.limit)


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-A OSINT collection")
    p.add_argument("--datasets", default="kp,dp,pp",
                   help="comma-separated tags from {kp,dp,pp}")
    p.add_argument("--limit", type=int, default=0,
                   help="max domains per dataset (0 = all). Useful for smoke tests.")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

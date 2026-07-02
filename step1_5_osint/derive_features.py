"""derive_features.py — raw OSINT cache → ML-ready typed features.

Pure Python, no network calls. Reads the gzipped cache produced by
`collect_osint.py` and emits per-dataset CSVs ready to be merged into
the Z matrices.

Every feature is Tier A (zero or near-zero leakage). Each one is also
listed in `FEATURE_TIERS` so the manifest records its provenance.

Sentinel values
───────────────
 -1   → unknown / unavailable / source error
  0   → factual zero (e.g. zero snapshots, zero certs)

Never NaN — keeps the downstream sklearn pipeline happy and makes the
manifest readable.

Usage
─────
    python reboot/step1_5_osint/derive_features.py --datasets kp,dp,pp

Outputs
───────
    reboot/runs/osint/osint_features_{tag}.csv
    reboot/runs/osint/feature_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
CACHE_ROOT = THIS.parent / "cache"
RAW_DIR = CACHE_ROOT / "raw"
TRANCO_FILE = CACHE_ROOT / "tranco" / "top1m.csv"

OSINT_RUNS = REBOOT / "runs" / "osint"

DATASET_JSONS = {
    "kp": REBOOT / "datasets" / "knowphish_data.json",
    "dp": REBOOT / "datasets" / "deltaphish_data.json",
    "pp": REBOOT / "datasets" / "phreshphish_data.json",
}

# Feature → leakage tier (A=zero, B=controlled, C=excluded). For audit
# completeness everything here is A; B/C lists kept for future ablations.
FEATURE_TIERS: dict[str, str] = {
    # Tranco
    "is_in_tranco_1m":           "A",
    "tranco_rank":               "A",
    "tranco_rank_log":           "A",
    # Wayback
    "wayback_first_seen_days":   "A",
    "wayback_last_seen_days":    "A",
    "wayback_n_snapshots":       "A",
    "wayback_active_window_days":"A",
    "wayback_first_status_ok":   "A",
    # CT (crt.sh analytics, beyond simple "first cert")
    "cert_history_count":        "A",
    "cert_first_seen_days":      "A",
    "cert_last_seen_days":       "A",
    "cert_issuer_diversity":     "A",
    "cert_is_free_only":         "A",
    "cert_has_wildcard":         "A",
    "cert_validity_median_days": "A",
    "cert_san_max":              "A",
    # RDAP
    "rdap_domain_age_days":      "A",
    "rdap_age_risk":             "A",
    "rdap_has_registrar":        "A",
    # DNS / BGP
    "dns_resolved":              "A",
    "asn_number":                "A",
    "asn_age_days":              "A",
    "asn_country":               "A",
}

FREE_ISSUERS_LOWER = [
    "let's encrypt", "letsencrypt", " r3", " r10", " r11", " e1", " e5",
    "zerossl", "buypass", "google trust services",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("derive_features")


# ── Utilities ─────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str | None) -> datetime | None:
    """Robust ISO-8601 parser. Returns a tz-aware UTC datetime or None.

    crt.sh emits naive datetimes (e.g. '2024-04-15T14:32:11'); we
    interpret those as UTC (the convention crt.sh follows internally).
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_wayback_ts(ts: str) -> datetime | None:
    """Wayback timestamps look like '20180412035021' (YYYYmmddHHMMSS)."""
    if not ts or len(ts) < 8:
        return None
    try:
        return datetime.strptime(ts[:14].ljust(14, "0"),
                                 "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _cache_load(source: str, key: str) -> dict | None:
    safe = key.replace("/", "_")[:200] or "_empty"
    p = RAW_DIR / source / f"{safe}.json.gz"
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── Tranco index ─────────────────────────────────────────────────


def load_tranco_index() -> dict[str, int]:
    """Load the Tranco list into {domain: rank}. ~10MB → ~80MB in memory."""
    idx: dict[str, int] = {}
    if not TRANCO_FILE.exists():
        log.warning(f"Tranco file missing at {TRANCO_FILE} — all tranco features will be -1")
        return idx
    with TRANCO_FILE.open() as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                rank, dom = parts
                try:
                    idx[dom.strip().lower()] = int(rank)
                except ValueError:
                    pass
    log.info(f"Tranco index: {len(idx):,} domains")
    return idx


def tranco_lookup(idx: dict[str, int], domain: str) -> tuple[int, int]:
    """Returns (is_in_1m, rank). Tries the domain and its registrable parent."""
    if not idx:
        return 0, -1
    d = domain.lower()
    if d in idx:
        return 1, idx[d]
    parts = d.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        parent = ".".join(parts[-3:])
    else:
        parent = ".".join(parts[-2:]) if len(parts) >= 2 else d
    if parent in idx:
        return 1, idx[parent]
    return 0, -1


# ── Per-source derivers ──────────────────────────────────────────


def derive_tranco(domain: str, idx: dict[str, int]) -> dict:
    is_in, rank = tranco_lookup(idx, domain)
    if rank > 0:
        rank_log = math.log10(rank)
    else:
        rank_log = -1.0
    return {
        "is_in_tranco_1m": is_in,
        "tranco_rank": rank,
        "tranco_rank_log": rank_log,
    }


def derive_wayback(domain: str) -> dict:
    """Snapshot timeline → first-seen, last-seen, count, active window."""
    cache = _cache_load("wayback", domain) or {}
    if not cache.get("ok"):
        return {k: -1 for k in (
            "wayback_first_seen_days", "wayback_last_seen_days",
            "wayback_n_snapshots", "wayback_active_window_days",
            "wayback_first_status_ok",
        )}
    snaps = cache.get("snapshots") or []
    if not snaps:
        return {
            "wayback_first_seen_days": -1,
            "wayback_last_seen_days": -1,
            "wayback_n_snapshots": 0,
            "wayback_active_window_days": -1,
            "wayback_first_status_ok": -1,
        }
    timestamps = [_parse_wayback_ts(row[0]) for row in snaps if row and row[0]]
    timestamps = [t for t in timestamps if t is not None]
    if not timestamps:
        return {
            "wayback_first_seen_days": -1,
            "wayback_last_seen_days": -1,
            "wayback_n_snapshots": len(snaps),
            "wayback_active_window_days": -1,
            "wayback_first_status_ok": -1,
        }
    now = _now()
    first = min(timestamps)
    last = max(timestamps)
    # First-capture HTTP status. Wayback rows: [timestamp, original, statuscode, length]
    first_idx = min(range(len(snaps)),
                    key=lambda i: snaps[i][0] if snaps[i] and snaps[i][0] else "9")
    first_status = (snaps[first_idx][2] if len(snaps[first_idx]) > 2 else "") or ""
    first_status_ok = 1 if first_status.startswith("2") else 0
    return {
        "wayback_first_seen_days": (now - first).days,
        "wayback_last_seen_days": (now - last).days,
        "wayback_n_snapshots": len(snaps),
        "wayback_active_window_days": (last - first).days,
        "wayback_first_status_ok": first_status_ok,
    }


def derive_ct_analytics(domain: str) -> dict:
    """crt.sh certificate analytics (Tier A — pure CT log data)."""
    cache = _cache_load("crtsh", domain) or {}
    if not cache.get("ok"):
        return {k: -1 for k in (
            "cert_history_count", "cert_first_seen_days",
            "cert_last_seen_days", "cert_issuer_diversity",
            "cert_is_free_only", "cert_has_wildcard",
            "cert_validity_median_days", "cert_san_max",
        )}
    certs = cache.get("certs") or []
    if not certs:
        return {
            "cert_history_count": 0,
            "cert_first_seen_days": -1,
            "cert_last_seen_days": -1,
            "cert_issuer_diversity": 0,
            "cert_is_free_only": -1,
            "cert_has_wildcard": -1,
            "cert_validity_median_days": -1,
            "cert_san_max": -1,
        }
    now = _now()
    starts, ends, issuers = [], [], []
    has_wildcard = 0
    san_counts: list[int] = []
    free_count, total_count = 0, 0
    for c in certs:
        nb = _parse_iso(c.get("not_before"))
        na = _parse_iso(c.get("not_after"))
        if nb:
            starts.append(nb)
        if na and nb:
            ends.append((na - nb).days)
        iss = (c.get("issuer_name") or "").lower()
        if iss:
            issuers.append(iss)
            total_count += 1
            if any(fi in f" {iss}" for fi in FREE_ISSUERS_LOWER):
                free_count += 1
        sans = (c.get("name_value") or "").split("\n")
        sans = [s.strip().lower() for s in sans if s.strip()]
        if any(s.startswith("*.") for s in sans):
            has_wildcard = 1
        san_counts.append(len(sans))
    if starts:
        first_seen_days = (now - min(starts)).days
        last_seen_days = (now - max(starts)).days
    else:
        first_seen_days = -1
        last_seen_days = -1
    issuer_set = set(issuers)
    validity_med = -1
    if ends:
        ends_sorted = sorted(ends)
        validity_med = ends_sorted[len(ends_sorted) // 2]
    is_free_only = -1
    if total_count:
        is_free_only = 1 if free_count == total_count else 0
    return {
        "cert_history_count": len(certs),
        "cert_first_seen_days": first_seen_days,
        "cert_last_seen_days": last_seen_days,
        "cert_issuer_diversity": len(issuer_set),
        "cert_is_free_only": is_free_only,
        "cert_has_wildcard": has_wildcard,
        "cert_validity_median_days": validity_med,
        "cert_san_max": max(san_counts) if san_counts else -1,
    }


def derive_rdap(domain: str) -> dict:
    """RDAP → registration date + registrar presence flag."""
    cache = _cache_load("rdap", domain) or {}
    if not cache.get("ok"):
        return {
            "rdap_domain_age_days": -1,
            "rdap_age_risk": -1,
            "rdap_has_registrar": -1,
        }
    data = cache.get("data") or {}
    created_str = None
    for event in (data.get("events") or []):
        if event.get("eventAction") == "registration":
            created_str = event.get("eventDate")
            break
    age_days = -1
    age_risk = -1
    if created_str:
        created = _parse_iso(created_str)
        if created:
            age_days = (_now() - created).days
            if age_days < 30:
                age_risk = 3
            elif age_days < 365:
                age_risk = 2
            else:
                age_risk = 1
    has_registrar = 0
    for entity in (data.get("entities") or []):
        roles = entity.get("roles") or []
        if "registrar" in roles:
            has_registrar = 1
            break
    return {
        "rdap_domain_age_days": age_days,
        "rdap_age_risk": age_risk,
        "rdap_has_registrar": has_registrar,
    }


def _ripestat_whois_field(records: list, key: str) -> str:
    """RIPE Stat WHOIS returns a list of records (one per RIR view).
    Each record is a list of {key, value, ...} dicts. We scan the
    union for the first matching key — different RIRs use different
    casings (RegDate vs registered), so we case-insensitive-match."""
    target = key.lower()
    for rec in records or []:
        for entry in rec or []:
            if (entry.get("key") or "").lower() == target:
                val = entry.get("value")
                if val:
                    return str(val)
    return ""


def derive_dns_bgp(domain: str) -> dict:
    """DNS resolution + RIPE Stat ASN allocation date."""
    dns_cache = _cache_load("dns", domain) or {}
    resolved = 1 if dns_cache.get("ok") and dns_cache.get("ips") else 0
    ip = (dns_cache.get("ips") or [None])[0]
    asn_number, asn_age_days, asn_country = -1, -1, ""
    if ip:
        net_cache = _cache_load("ripestat_network", ip) or {}
        if net_cache.get("ok"):
            asns = ((net_cache.get("data") or {}).get("data") or {}).get("asns") or []
            for a in asns:
                try:
                    asn_number = int(str(a).removeprefix("AS"))
                    break
                except (ValueError, AttributeError):
                    continue
    if asn_number > 0:
        asn_cache = _cache_load("ripestat_asn", str(asn_number)) or {}
        if asn_cache.get("ok"):
            records = ((asn_cache.get("data") or {}).get("data") or {}).get("records") or []
            # RegDate is ARIN's name for the allocation date.
            # Other RIRs use "created" or "registered".
            for key_candidate in ("RegDate", "created", "registered"):
                raw = _ripestat_whois_field(records, key_candidate)
                if raw:
                    d = _parse_iso(raw if "T" in raw else raw + "T00:00:00")
                    if d:
                        asn_age_days = (_now() - d).days
                        break
            asn_country = (
                _ripestat_whois_field(records, "country")
                or _ripestat_whois_field(records, "Country")
                or ""
            )
    return {
        "dns_resolved": resolved,
        "asn_number": asn_number,
        "asn_age_days": asn_age_days,
        "asn_country": asn_country,
    }


# ── Per-dataset assembly ─────────────────────────────────────────


def domains_for_dataset(tag: str) -> list[tuple[int, str, int]]:
    """Return list of (sample_id, domain, label) from the dataset JSON."""
    p = DATASET_JSONS[tag]
    with p.open() as f:
        entries = json.load(f)
    out: list[tuple[int, str, int]] = []
    for e in entries:
        domain = (e.get("domain_name") or "").strip().lower()
        sid = e.get("id")
        lbl = e.get("label")
        if not domain or sid is None:
            continue
        out.append((int(sid), domain, int(lbl) if lbl is not None else -1))
    return out


def derive_for_domain(domain: str, tranco_idx: dict[str, int]) -> dict:
    row: dict = {"domain": domain}
    row.update(derive_tranco(domain, tranco_idx))
    row.update(derive_wayback(domain))
    row.update(derive_ct_analytics(domain))
    row.update(derive_rdap(domain))
    row.update(derive_dns_bgp(domain))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="kp,dp,pp")
    args = parser.parse_args()

    OSINT_RUNS.mkdir(parents=True, exist_ok=True)
    tranco_idx = load_tranco_index()

    feature_cols = [
        "is_in_tranco_1m", "tranco_rank", "tranco_rank_log",
        "wayback_first_seen_days", "wayback_last_seen_days",
        "wayback_n_snapshots", "wayback_active_window_days",
        "wayback_first_status_ok",
        "cert_history_count", "cert_first_seen_days", "cert_last_seen_days",
        "cert_issuer_diversity", "cert_is_free_only", "cert_has_wildcard",
        "cert_validity_median_days", "cert_san_max",
        "rdap_domain_age_days", "rdap_age_risk", "rdap_has_registrar",
        "dns_resolved", "asn_number", "asn_age_days", "asn_country",
    ]

    # Manifest of feature → tier.
    (OSINT_RUNS / "feature_manifest.json").write_text(
        json.dumps({
            "feature_tiers": FEATURE_TIERS,
            "features_emitted": feature_cols,
            "tranco_size": len(tranco_idx),
            "generated_at_utc": _now().isoformat(),
        }, indent=2)
    )

    for tag in [t.strip() for t in args.datasets.split(",")]:
        if tag not in DATASET_JSONS:
            log.error(f"Unknown dataset {tag!r}")
            continue
        sample_rows = domains_for_dataset(tag)
        log.info(f"[{tag}] deriving features for {len(sample_rows)} samples "
                 f"({len({d for _, d, _ in sample_rows})} unique domains)")
        # Compute features per unique domain (caches are domain-keyed).
        cache: dict[str, dict] = {}
        for sid, domain, _ in sample_rows:
            if domain not in cache:
                cache[domain] = derive_for_domain(domain, tranco_idx)

        out_path = OSINT_RUNS / f"osint_features_{tag}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "domain", "label"] + feature_cols)
            for sid, domain, lbl in sample_rows:
                feats = cache.get(domain, {})
                w.writerow(
                    [sid, domain, lbl]
                    + [feats.get(c, -1) for c in feature_cols]
                )

        # Coverage summary for sanity.
        known_counts = {c: 0 for c in feature_cols}
        for sid, domain, _ in sample_rows:
            feats = cache.get(domain, {})
            for c in feature_cols:
                v = feats.get(c, -1)
                if c == "asn_country":
                    if v:
                        known_counts[c] += 1
                else:
                    if v not in (-1, -1.0):
                        known_counts[c] += 1
        log.info(f"[{tag}] coverage:")
        for c in feature_cols:
            log.info(f"  {c:<32s} {known_counts[c]}/{len(sample_rows)} "
                     f"= {known_counts[c]/max(len(sample_rows),1):.0%}")
        log.info(f"[{tag}] wrote {out_path}")


if __name__ == "__main__":
    main()

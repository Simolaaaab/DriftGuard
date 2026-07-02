# Step 1.5 — OSINT Enrichment

Goal: replace the narrow `code/CERT/` cert-only pipeline with a
broader Tier-A OSINT collection, producing a leakage-clean enriched
Z matrix for KnowPhish, DeltaPhish, and PhreshPhish.

The end of this step gives us three new files:
`reboot/runs/osint/z_matrix_{kp,dp,pp}_osint.csv`, ready to be fed
back into `step1_ensemble/ensemble_reboot.py` for the with-OSINT
baseline.

## Leakage tier policy

| Tier | Definition | Sources included |
|------|-----------|------------------|
| **A** | Zero or near-zero label leakage with our KP/DP/PP datasets. Sources are popularity / archival / DNS / registration / BGP — ortogonal to any phishing labeling. | **Tranco**, **Wayback Machine CDX**, **crt.sh CT logs**, **RDAP**, **DNS**, **RIPE Stat** |
| **B** | Controlled leakage. Could be biased if the source aggregator's training set overlaps our datasets. Used only for ablation studies in this paper. | Domain-registrar reputation aggregators, ASN abuse rate tables |
| **C** | Structural leakage. The source's labels are likely correlated with our ground truth. Never enters the ensemble; can appear in the LLM oracle prompt as an *advisory hint*, never as a feature column. | VirusTotal, PhishTank, OpenPhish, GSB, AbuseIPDB, URLhaus, ThreatFox, Spamhaus |

This step implements **Tier A only**. The B/C lists exist in the
manifest so the paper can describe the methodology completely; we
won't query them.

## Sources implemented (Tier A)

| Source | Free? | Key required? | Used for | Cache key |
|--------|-------|---------------|----------|-----------|
| Tranco top-1M | ✓ | no | popularity rank | offline file |
| Wayback CDX API | ✓ | no | first/last archive timeline | domain |
| crt.sh | ✓ | no | CT issuance history | domain |
| RDAP (rdap.org) | ✓ | no | registration date + registrar | registrable domain |
| System DNS | ✓ | no | resolve → IP for BGP chain | domain |
| RIPE Stat network-info | ✓ | no | IP → ASN | IP |
| RIPE Stat whois | ✓ | no | ASN → allocation date | ASN number |

Every response is persisted as a gzipped JSON in
`reboot/step1_5_osint/cache/raw/<source>/<key>.json.gz` with
`fetched_at_utc` metadata for reproducibility. Re-running is a no-op
where the cache is warm.

## Features produced

See `feature_manifest.json` for the authoritative list. As of the
current implementation:

```
TRANCO  (3)
  is_in_tranco_1m        : 0/1
  tranco_rank            : 1..1_000_000 or -1
  tranco_rank_log        : log10 of rank, or -1

WAYBACK (5)
  wayback_first_seen_days   : days since first archive
  wayback_last_seen_days    : days since last archive
  wayback_n_snapshots       : snapshot count (cap 1000)
  wayback_active_window_days: last - first, in days
  wayback_first_status_ok   : 1 if first capture was HTTP 2xx, else 0/-1

CT      (8)  (zero-leakage analytics of the CT log)
  cert_history_count        : total certs ever logged
  cert_first_seen_days      : days since oldest cert not_before
  cert_last_seen_days       : days since most recent cert not_before
  cert_issuer_diversity     : count of distinct issuers
  cert_is_free_only         : 1 if all certs are Let's Encrypt / ZeroSSL / Buypass
  cert_has_wildcard         : 1 if any cert SAN starts with '*.'
  cert_validity_median_days : median validity period across history
  cert_san_max              : largest SAN list observed

RDAP    (3)
  rdap_domain_age_days      : age in days, -1 if unknown (ccTLD/GDPR)
  rdap_age_risk             : 1=low (>1y), 2=mid (30d-1y), 3=high (<30d), -1=?
  rdap_has_registrar        : 1 if RDAP carries a registrar entity, else 0/-1

DNS/BGP (4)
  dns_resolved              : 1 if domain resolves today
  asn_number                : current ASN, -1 if unknown
  asn_age_days              : days since ASN was first allocated
  asn_country               : country code (string, not numeric)
```

23 features total. Every numeric feature uses the sentinel `-1` for
"unknown / source error". Never NaN.

## Running

```bash
# 0. one-time prep — synth KP JSON (already done if you ran step1)
python reboot/step1_ensemble/prepare_kp_json.py

# 1. async OSINT collection — long-running, resumable, cache-safe
python reboot/step1_5_osint/collect_osint.py --datasets kp,dp,pp

# 2. derive features from the cache (fast, pure Python)
python reboot/step1_5_osint/derive_features.py --datasets kp,dp,pp

# 3. merge into Z matrices
python reboot/step1_5_osint/enrich_z_matrices.py

# 4. re-run Step 1 ensemble with OSINT features (re-uses ensemble_reboot.py)
python reboot/step1_ensemble/ensemble_reboot.py
```

`ensemble_reboot.py` auto-detects the new OSINT columns in the Z
matrices and will include them in the feature set. For the official
comparison run we'll wire a `--mode {classic,osint}` flag — for now,
swap the Z paths if you want a clean classic-vs-osint A/B.

## Resumption & re-fetching

`collect_osint.py` skips any (source, key) where the cache file
exists. To force a re-fetch of one source for all domains:

```bash
rm -rf reboot/step1_5_osint/cache/raw/wayback
python reboot/step1_5_osint/collect_osint.py --datasets kp,dp,pp
```

Per-domain re-fetch:

```bash
rm reboot/step1_5_osint/cache/raw/crtsh/example.com.json.gz
python reboot/step1_5_osint/collect_osint.py --datasets pp --limit 1
```

## Expected timings

Rate limits are intentionally conservative (we're guests on free APIs):

| Source | Concurrency | Min interval | Effective rate |
|--------|------------|--------------|----------------|
| crt.sh | 2 | 1.0 s | ~2/s |
| Wayback CDX | 4 | 0.3 s | ~13/s |
| RDAP | 2 | 0.6 s | ~3/s |
| RIPE Stat network-info | 6 | 0.15 s | ~40/s |
| RIPE Stat whois | 6 | 0.15 s | ~40/s |

Per-domain wall-clock is roughly bounded by crt.sh — expect ~1.5s
per domain *with full cache miss*. With concurrency the throughput
on a 5000-domain set is ~30–60 minutes. Subsequent re-runs are
near-instant (everything cached).

## Why we excluded Passive DNS from Tier A (decision log)

The truly useful passive-DNS providers (DNSDB Farsight, SecurityTrails
paid, RiskIQ) are not free at the dataset sizes we need. The
remaining free options (SecurityTrails 50/mo free, CIRCL PDNS with
auth, Mnemonic with auth) either rate-limit too aggressively or
require accounts. We chose to use Wayback Machine + DNS + RIPE Stat as
a triad that approximates the same lifecycle signals (first-seen,
last-seen, churn, hosting movement) without any account.

If the user later wants to integrate paid passive DNS, the
`sources.py` module is the natural extension point — add a
`fetch_passivedns(client, domain)` returning the same dict shape and
wire it into `collect_one_domain` in `collect_osint.py`.

## Dependencies

```
httpx>=0.27        # async HTTP client (with HTTP/2)
pandas>=2.0
```

`pip install httpx` is the only new dep beyond what `step1_ensemble`
already needs.

## Provenance & reproducibility

Every cache file records `fetched_at_utc`. The feature manifest also
records `generated_at_utc` and the size of the Tranco snapshot used.
This is what gets versioned with the paper supplement.

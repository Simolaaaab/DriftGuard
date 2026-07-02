"""enrich_z_matrices.py — Merge OSINT features into the three Z matrices.

Reads `reboot/runs/osint/osint_features_{tag}.csv` (produced by
`derive_features.py`) and joins it onto the existing Z matrix on
`sample_id` first, falling back to a domain match for KP (where the
sample_id is synthetic).

Output: `reboot/runs/osint/z_matrix_{tag}_osint.csv` for each tag.

Provenance is preserved: every OSINT column keeps its sentinel value
(-1) when the join fails, so downstream sklearn can use mean
imputation on a stable signal.

Usage
─────
    python reboot/step1_5_osint/enrich_z_matrices.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

REBOOT = Path(__file__).resolve().parent.parent

Z_PATHS = {
    "kp": REBOOT / "datasets" / "z_matrix_knowphish.csv",
    "dp": REBOOT / "datasets" / "z_matrix_spacephish.csv",
    "pp": REBOOT / "datasets" / "z_matrix_phreshphish.csv",
}
OSINT_RUNS = REBOOT / "runs" / "osint"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("enrich_z_matrices")


def kp_domain_from_sample_name(name: str) -> str:
    """KP sample_name format: '+<domain>+<date>+<seq>'.
    Matches `prepare_kp_json.py` exactly so the join is consistent."""
    s = str(name).strip()
    if s.startswith("+"):
        s = s[1:]
    if "+" in s:
        return s.split("+", 1)[0].lower()
    if "_" in s:
        return s.split("_", 1)[0].lower()
    return s.lower()


def dp_pp_domain_from_sample_name(name: str) -> str:
    """DP/PP sample_name: '<id>_<host>'. Take everything after the first
    underscore. Lowercased."""
    s = str(name).strip()
    if "_" in s:
        return s.split("_", 1)[1].lower()
    return s.lower()


def join_one(tag: str) -> None:
    z_path = Z_PATHS[tag]
    osint_path = OSINT_RUNS / f"osint_features_{tag}.csv"
    if not osint_path.exists():
        log.warning(f"[{tag}] OSINT CSV missing: {osint_path} — skipping")
        return
    df_z = pd.read_csv(z_path)
    df_o = pd.read_csv(osint_path)
    log.info(f"[{tag}] Z={df_z.shape}  OSINT={df_o.shape}")

    osint_cols = [c for c in df_o.columns if c not in ("sample_id", "label", "domain")]

    # First attempt: join on sample_id when both sides expose one.
    # DP/PP have integer sample_id; KP doesn't (synthesised at JSON-prep
    # time). The OSINT CSV always has the synthetic sample_id used by
    # the corresponding JSON.
    merged = None
    if "sample_id" in df_z.columns:
        merged = df_z.merge(
            df_o[["sample_id"] + osint_cols],
            on="sample_id", how="left",
        )
        n_matched = merged[osint_cols[0]].notna().sum()
        log.info(f"[{tag}] join-by-sample_id matched {n_matched}/{len(df_z)}")
        if n_matched == 0:
            merged = None

    # Fallback: join on domain.
    if merged is None:
        if tag == "kp":
            df_z["__join_domain"] = df_z["sample_name"].map(kp_domain_from_sample_name)
        else:
            df_z["__join_domain"] = df_z["sample_name"].map(dp_pp_domain_from_sample_name)
        # OSINT CSV exposes 'domain' explicitly.
        df_o_unique = df_o.drop_duplicates(subset="domain", keep="first")
        merged = df_z.merge(
            df_o_unique[["domain"] + osint_cols],
            left_on="__join_domain", right_on="domain", how="left",
            suffixes=("", "_osint"),
        )
        merged = merged.drop(columns=["__join_domain", "domain"], errors="ignore")
        n_matched = merged[osint_cols[0]].notna().sum()
        log.info(f"[{tag}] join-by-domain matched {n_matched}/{len(df_z)}")

    # Fill unmatched cells with -1 sentinel.
    for c in osint_cols:
        if c == "asn_country":
            merged[c] = merged[c].fillna("")
        else:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(-1)

    out_path = OSINT_RUNS / f"z_matrix_{tag}_osint.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"[{tag}] wrote {out_path}  shape={merged.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="kp,dp,pp")
    args = parser.parse_args()
    OSINT_RUNS.mkdir(parents=True, exist_ok=True)
    for tag in [t.strip() for t in args.datasets.split(",") if t.strip()]:
        if tag not in Z_PATHS:
            log.error(f"Unknown dataset {tag!r}")
            continue
        join_one(tag)


if __name__ == "__main__":
    main()

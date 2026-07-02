"""prepare_kp_json.py — synth KnowPhish JSON for the OSINT pipeline.

`code/CERT/collect_certificates.py` and the new `step1_5_osint/`
collector both consume a JSON file shaped like
`phishhook-agent/datasets/{deltaphish,phreshphish}_data.json`:

    [{"id": <int>, "url": "<full URL>", "domain_name": "<host>",
      "label": 0|1}, ...]

KnowPhish samples in `z_matrix_knowphish.csv` only carry `sample_name`
in the form `+<domain>+<YYYY_MM_DD>+<seq>` — no full URL. The OSINT
sources (crt.sh, RDAP, Wayback, BGP, Tranco) all key on the domain
itself, so we build a synthetic `https://<domain>/` URL.

Run once before launching collection:
    python reboot/step1_ensemble/prepare_kp_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REBOOT = Path(__file__).resolve().parent.parent
Z_KP = REBOOT / "datasets" / "z_matrix_knowphish.csv"
OUT = REBOOT / "datasets" / "knowphish_data.json"


def kp_domain_from_sample_name(name: str) -> str:
    """KP sample_name format: '+<domain>+<date>+<seq>'."""
    s = str(name).strip()
    if s.startswith("+"):
        s = s[1:]
    if "+" in s:
        return s.split("+", 1)[0].lower()
    if "_" in s:
        return s.split("_", 1)[0].lower()
    return s.lower()


def main() -> None:
    df = pd.read_csv(Z_KP)
    print(f"KP rows: {len(df)}")

    seen: set[str] = set()
    entries: list[dict] = []
    for i, row in df.iterrows():
        domain = kp_domain_from_sample_name(row["sample_name"])
        if not domain or domain in seen:
            continue
        seen.add(domain)
        entries.append({
            "id": int(i),
            "url": f"https://{domain}/",
            "domain_name": domain,
            "label": int(row["label"]),
            "sample_name": str(row["sample_name"]),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(entries, f)
    print(f"Wrote {len(entries)} unique-domain entries to {OUT.relative_to(REBOOT)}")

    n_phish = sum(1 for e in entries if e["label"] == 1)
    n_benign = sum(1 for e in entries if e["label"] == 0)
    print(f"  Label balance: phish={n_phish}  benign={n_benign}")
    print(f"\nOSINT collection ETA at ~2s/domain async: "
          f"{len(entries) * 2.0 / 3600:.1f} hours")


if __name__ == "__main__":
    main()

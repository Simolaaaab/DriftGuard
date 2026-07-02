"""fig4_oracle_health.py — Per-round oracle health: DeepSeek vs Kimi.

Three side-by-side panels (parse_ok, class_skew, yield) with two
strategies overlaid (DeepSeek-V4-Flash healthy, Kimi-K2.6 broken).
Threshold lines marked. The Kimi curves cross all critical thresholds
from round 1; DeepSeek never does.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import COLORS, TWO_COL, save

# Allow importing common types for label parsing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step3_al"))
from common import OracleLabel
from oracle_health import HealthMonitor, HealthThresholds

AL_RUNS = Path(__file__).resolve().parents[1] / "runs" / "al"


def _llm_to_int(s):
    s = str(s).strip().lower()
    if s in ("phish", "phishing", "1"): return 1
    if s in ("benign", "legitimate", "0"): return 0
    return None


def replay_health(run_id: str) -> list:
    """Replay completed run through HealthMonitor, return per-round audits."""
    rf = AL_RUNS / run_id / "rounds.jsonl"
    cd = AL_RUNS / run_id / "oracle_cache" / "B"
    rounds = [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
    mon = HealthMonitor()
    audits = []
    for rec in rounds:
        labels = []
        for sid in rec.get("queried_sample_ids", []):
            p = cd / f"{sid}.json.gz"
            if not p.exists():
                p2 = Path(f"reboot/runs/oracle_ablation/cache/D_html_osint/{sid}.json.gz")
                if p2.exists(): p = p2
                else:
                    labels.append(OracleLabel(sample_id=sid, label=None, confidence=0.0,
                                              reasoning="", source="missing", raw=None))
                    continue
            try:
                with gzip.open(p, "rt") as f: d = json.load(f)
            except Exception:
                labels.append(OracleLabel(sample_id=sid, label=None, confidence=0.0,
                                          reasoning="", source="parse_err", raw=None))
                continue
            if not d.get("ok"):
                labels.append(OracleLabel(sample_id=sid, label=None, confidence=0.0,
                                          reasoning="", source="not_ok", raw=None))
                continue
            labels.append(OracleLabel(
                sample_id=sid, label=_llm_to_int(d.get("label_llm")),
                confidence=float(d.get("confidence") or 0.0),
                reasoning="", source="llm", raw=d,
            ))
        n_a = sum(1 for x in labels if x.label is not None)
        audits.append(mon.tick(rec["round"], labels, n_a))
    return audits, mon


def main() -> None:
    th = HealthThresholds()
    audits_ds, mon_ds = replay_health("drift_anchored_llm_periodic_temporal_B")
    audits_ki, mon_ki = replay_health("drift_anchored_B_kimi_seed2025")

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL * 1.15, 2.6), sharex=True)
    x_ds = [a.round_num for a in audits_ds]
    x_ki = [a.round_num for a in audits_ki]

    # ── parse_ok ──
    ax = axes[0]
    ax.plot(x_ds, [a.parse_ok_rate for a in audits_ds],
            color="#2E86AB", lw=1.6, marker="o", markersize=2.5,
            label="DeepSeek-V4-Flash")
    ax.plot(x_ki, [a.parse_ok_rate for a in audits_ki],
            color="#E84855", lw=1.6, marker="s", markersize=2.5,
            label="Kimi-K2.6")
    ax.axhline(th.parse_ok_critical, color="red", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.5, th.parse_ok_critical - 0.04, "CRITICAL",
            fontsize=7, color="red")
    ax.axhline(th.parse_ok_warn, color="orange", lw=0.6, ls=":", alpha=0.6)
    ax.set_title("(a) JSON parse-OK rate")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    # ── class_skew ──
    ax = axes[1]
    ax.plot(x_ds, [a.class_skew for a in audits_ds],
            color="#2E86AB", lw=1.6, marker="o", markersize=2.5)
    ax.plot(x_ki, [a.class_skew for a in audits_ki],
            color="#E84855", lw=1.6, marker="s", markersize=2.5)
    ax.axhline(th.class_skew_critical, color="red", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.5, th.class_skew_critical + 0.01, "CRITICAL",
            fontsize=7, color="red")
    ax.axhline(th.class_skew_warn, color="orange", lw=0.6, ls=":", alpha=0.6)
    ax.set_title("(b) Class skew  ↓ lower is healthier")
    ax.set_xlabel("AL round")
    ax.set_ylim(0.45, 1.05)

    # ── yield ──
    ax = axes[2]
    ax.plot(x_ds, [a.yield_rate for a in audits_ds],
            color="#2E86AB", lw=1.6, marker="o", markersize=2.5)
    ax.plot(x_ki, [a.yield_rate for a in audits_ki],
            color="#E84855", lw=1.6, marker="s", markersize=2.5)
    ax.axhline(th.yield_critical, color="red", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.5, th.yield_critical - 0.04, "CRITICAL",
            fontsize=7, color="red")
    ax.axhline(th.yield_warn, color="orange", lw=0.6, ls=":", alpha=0.6)
    ax.set_title("(c) Yield rate (accepted/queried)")
    ax.set_ylim(0, 1.05)

    # HALT marker for Kimi
    halt_at = next((a.round_num for a in audits_ki
                    if a.signal.value == "critical"), None)
    if halt_at and halt_at < 25:
        for ax in axes:
            ax.axvline(halt_at + 1, color="red", lw=0.8, ls=":", alpha=0.4)
            ax.text(halt_at + 1.2, ax.get_ylim()[1] * 0.94,
                    "HALT" if ax == axes[0] else "",
                    fontsize=8, color="red", weight="bold")

    fig.suptitle("Label-free oracle health timeline — DeepSeek-V4-Flash vs "
                 "Kimi-K2.6 (drift_anchored, K=50, T=25 rounds)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, "fig4_oracle_health")
    plt.close(fig)


if __name__ == "__main__":
    main()

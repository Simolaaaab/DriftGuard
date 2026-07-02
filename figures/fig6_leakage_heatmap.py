"""fig6_leakage_heatmap.py — Per-feature AUC-alone heatmap × dataset.

Visualises the Step-1 leakage gate: features that score high on KP
but low on DP/PP (left column tall, right columns short) are gate-2
rejects. Sorted by KP AUC desc. Gate decisions overlaid as marker
column on the right.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import COLORS, ONE_COL, save

REBOOT = Path(__file__).resolve().parents[1]
AUC_CSV = REBOOT / "runs" / "ensemble" / "osint" / "feature_auc_per_dataset.csv"
MANIFEST = REBOOT / "runs" / "ensemble" / "osint" / "feature_manifest.json"


def main() -> None:
    df = pd.read_csv(AUC_CSV)
    # CSV is already wide: columns = feature, dp, kp, pp.
    df = df.set_index("feature")
    pivot = df[["kp", "dp", "pp"]]
    # Drop rows that are NaN on all three (shouldn't happen but be safe)
    pivot = pivot.dropna(how="all")
    # Sort by KP desc to put leaky on top
    pivot = pivot.sort_values("kp", ascending=False)

    # Load gate manifest to mark rejected features
    manifest = json.loads(MANIFEST.read_text())
    rejected = {r["feature"] for r in manifest["rejected_features"]}
    kp_only = set(manifest.get("kp_only_features_dropped", []))

    fig, ax = plt.subplots(figsize=(ONE_COL * 2.0, 8.5))
    # Use fillna for display (DP/PP missing → grey)
    mat = pivot.values
    mat_display = np.where(np.isnan(mat), -1, mat)
    cmap = plt.get_cmap("RdYlGn_r")  # red=high AUC=suspicious, green=low
    im = ax.imshow(mat_display, aspect="auto", cmap=cmap, vmin=0.45, vmax=1.0)

    # Mark NaN cells with grey overlay
    nan_mask = np.isnan(mat)
    for i in range(nan_mask.shape[0]):
        for j in range(nan_mask.shape[1]):
            if nan_mask[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#dddddd", edgecolor="none"))

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                t = "—"; color = "#666"
            else:
                t = f"{v:.2f}"
                color = "white" if v > 0.85 or v < 0.55 else "black"
            ax.text(j, i, t, ha="center", va="center",
                    fontsize=7, color=color)

    # Mark gate verdict on the right margin
    for i, feat in enumerate(pivot.index):
        if feat in rejected:
            marker = "✗"; color = "red"; offset = 3.6
        elif feat in kp_only:
            marker = "↪"; color = "#888"; offset = 3.6
        else:
            marker = "✓"; color = "#118C5C"; offset = 3.6
        ax.text(offset, i, marker, ha="left", va="center",
                fontsize=10, color=color, weight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["KP", "DP", "PP"], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7.5)
    ax.set_xlim(-0.5, 4.5)

    # Title + caption
    ax.set_title("Per-feature AUC-alone across datasets\n"
                 "(✗ = rejected by gate-1/-2, ↪ = KP-only dropped, ✓ = kept)",
                 fontsize=9, pad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.16)
    cbar.set_label("AUC-alone", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "fig6_leakage_heatmap")
    plt.close(fig)


if __name__ == "__main__":
    main()

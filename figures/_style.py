"""Shared style for paper figures.

Publication-quality settings, paper-friendly palette, vector PDF output.
Each figure script imports from here and never sets rcParams itself.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Vector output, embed fonts properly for camera-ready submissions.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# Colour palette — colour-blind friendly. Anchored per concept so
# the same idea wears the same colour across figures.
COLORS = {
    "pp":              "#2E86AB",  # blue — natural drift recovery
    "kp":              "#A23B72",  # plum — in-distribution
    "dp":              "#E84855",  # red — adversarial
    "gt":              "#000000",  # black — ground truth reference
    # Strategy palette
    "drift_anchored":  "#2E86AB",
    "margin":          "#E84855",
    "qbc":             "#7B5EA7",
    "hybrid":          "#118C5C",
    "core_set":        "#F2A65A",
    "badge":           "#5B5B5B",
    "random":          "#B0B0B0",
    # Oracle prompt variants
    "variant_d":       "#7B5EA7",
    "variant_b":       "#118C5C",
    # Health monitor states
    "healthy":         "#118C5C",
    "warn":            "#F2A65A",
    "critical":        "#E84855",
}

# Column sizing helpers (assume single-column = 3.5", full = 7")
ONE_COL = 3.5
TWO_COL = 7.0
HALF_COL = 1.7

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    """Save fig as both PDF (paper) and PNG (preview)."""
    p_pdf = OUT_DIR / f"{name}.pdf"
    p_png = OUT_DIR / f"{name}.png"
    fig.savefig(p_pdf)
    fig.savefig(p_png, dpi=150)
    print(f"  → {p_pdf.relative_to(OUT_DIR.parents[1])}")
    print(f"  → {p_png.relative_to(OUT_DIR.parents[1])}")

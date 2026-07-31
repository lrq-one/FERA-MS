from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Output paths
# -----------------------------
ROOT = Path("/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119")
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "Fig2D_component_ablation.png"
PDF_PATH = OUT_DIR / "Fig2D_component_ablation.pdf"


# -----------------------------
# Data
# Three-seed mean ± sample std
# -----------------------------
variants = [
    "Full\nFERA-MS",
    "Global-only\nCE",
    "w/o local\nm/z expansion",
    "w/o rendered-\nentry gate",
    "w/o candidate\nreranker",
    "w/o spectrum\nallocator",
]

micro_cbin_mean = np.array([
    0.655562,       # Full FERA-MS
    0.6164874832,   # Global ACE only
    0.589226,       # w/o local m/z expansion
    0.613290,       # w/o rendered-entry gate
    0.647067,       # w/o candidate reranker
    0.650640,       # w/o spectrum allocator
])

micro_cbin_std = np.array([
    0.005775,
    0.0031676190,
    0.006489,
    0.001465,
    0.005371,
    0.006474,
])

micro_jss_mean = np.array([
    0.621972,       # Full FERA-MS
    0.5664416750,   # Global ACE only
    0.551011,       # w/o local m/z expansion
    0.578175,       # w/o rendered-entry gate
    0.613919,       # w/o candidate reranker
    0.620225,       # w/o spectrum allocator
])

micro_jss_std = np.array([
    0.004732,
    0.0016327962,
    0.004917,
    0.002099,
    0.004572,
    0.005141,
])


# -----------------------------
# Style
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

COLOR_CBIN = "#3C82C4"   # Blue
COLOR_JSS = "#7E69AC"    # Purple
EDGE = "#2F2F2F"
GRID = "#D9DDE3"

fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=600)

group_spacing = 0.84
x = np.arange(len(variants)) * group_spacing
width = 0.34

error_kw = {
    "elinewidth": 0.9,
    "ecolor": EDGE,
    "capthick": 0.9,
}

bars1 = ax.bar(
    x - width / 2,
    micro_cbin_mean,
    width,
    yerr=micro_cbin_std,
    color=COLOR_CBIN,
    edgecolor=EDGE,
    linewidth=0.6,
    capsize=4,
    error_kw=error_kw,
    label="Micro CBIN",
    zorder=3,
)

bars2 = ax.bar(
    x + width / 2,
    micro_jss_mean,
    width,
    yerr=micro_jss_std,
    color=COLOR_JSS,
    edgecolor=EDGE,
    linewidth=0.6,
    capsize=4,
    error_kw=error_kw,
    label="Micro JSS",
    zorder=3,
)


# -----------------------------
# Grid and axes
# -----------------------------
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.xaxis.grid(False)

# Keep the truncated y-axis used in the original panel,
# while leaving enough space for error bars and labels.
ax.set_ylim(0.54, 0.682)
ax.set_yticks([0.54, 0.57, 0.60, 0.63, 0.66])

ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=10)
ax.set_ylabel("Similarity", fontsize=12)

ax.set_title(
    "D CE conditioning and component ablations",
    loc="left",
    fontsize=20,
    fontweight="bold",
    pad=10,
)


# -----------------------------
# Legend
# -----------------------------
ax.legend(
    frameon=False,
    ncol=2,
    loc="upper right",
    fontsize=10,
    handlelength=1.6,
    columnspacing=1.2,
)


# -----------------------------
# Emphasize the full model label
# -----------------------------
xticklabels = ax.get_xticklabels()
if xticklabels:
    xticklabels[0].set_fontweight("bold")


# -----------------------------
# Numeric labels above error bars
# -----------------------------
def annotate_bars(bars, values, errors):
    for bar, value, error in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
            zorder=5,
            clip_on=False,
        )


annotate_bars(bars1, micro_cbin_mean, micro_cbin_std)
annotate_bars(bars2, micro_jss_mean, micro_jss_std)


# -----------------------------
# Clean spines
# -----------------------------
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

ax.tick_params(axis="both", width=0.8)
ax.margins(x=0.025)


# -----------------------------
# Save
# -----------------------------
fig.tight_layout()

fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

plt.close(fig)

print(f"SAVED PNG: {PNG_PATH}")
print(f"SAVED PDF: {PDF_PATH}")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
# -----------------------------
variants = [
    "Full\nFERA-MS",
    "w/o local\nm/z expansion",
    "w/o rendered-\npeak gate",
    "w/o candidate\nreranker",
    "w/o spectrum\nallocator",
]

micro_cbin_mean = np.array([
    0.655562,
    0.589226,
    0.613290,
    0.647067,
    0.650640,
])

micro_cbin_std = np.array([
    0.005775,
    0.006489,
    0.001465,
    0.005371,
    0.006474,
])

micro_jss_mean = np.array([
    0.621972,
    0.551011,
    0.578175,
    0.613919,
    0.620225,
])

micro_jss_std = np.array([
    0.004732,
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

COLOR_CBIN = "#3C82C4"   # blue
COLOR_JSS  = "#7E69AC"   # purple
EDGE = "#2F2F2F"
GRID = "#D9DDE3"

fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)

x = np.arange(len(variants))
width = 0.34

bars1 = ax.bar(
    x - width / 2,
    micro_cbin_mean,
    width,
    yerr=micro_cbin_std,
    color=COLOR_CBIN,
    edgecolor=EDGE,
    linewidth=0.6,
    capsize=4,
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
    label="Micro JSS",
    zorder=3,
)

# Grid and axes
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.xaxis.grid(False)

# 给顶部多留一点空间，避免数字和误差棒挤在一起
ax.set_ylim(0.54, 0.678)
ax.set_yticks([0.61, 0.61, 0.61, 0.64, 0.67])

ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=10)
ax.set_ylabel("Similarity", fontsize=12)
ax.set_title("D  Key component ablations", loc="left", fontsize=18, fontweight="bold", pad=10)

# Legend
ax.legend(
    frameon=False,
    ncol=2,
    loc="upper right",
    fontsize=10,
    handlelength=1.6,
    columnspacing=1.2,
)

# 第一组加粗
xtls = ax.get_xticklabels()
if xtls:
    xtls[0].set_fontweight("bold")

# Numeric labels on bars
def annotate_bars(bars, values):
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,   # 往上提，避免压到误差棒
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
            zorder=5,
            clip_on=False,
        )

annotate_bars(bars1, micro_cbin_mean)
annotate_bars(bars2, micro_jss_mean)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
fig.savefig(PDF_PATH, bbox_inches="tight")
print(f"SAVED PNG: {PNG_PATH}")
print(f"SAVED PDF: {PDF_PATH}")

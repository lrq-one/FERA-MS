from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3B82C4"
BLUE2 = "#8DB9E2"
PURPLE = "#76639A"
PURPLE2 = "#B2A3D4"
INK = "#1E2A38"
GRID = "#D9DEE5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

labels = ["D1", "D2", "D3", "D3 + H", "D3 + H + NL"]
x = np.arange(len(labels))
w = 0.18

random_peak = np.array([4.596, 23.784, 37.521, 85.473, 92.707])
scaffold_peak = np.array([3.755, 22.841, 36.065, 83.521, 92.118])
random_int = np.array([12.988, 33.388, 46.679, 91.788, 96.269])
scaffold_int = np.array([9.244, 31.387, 44.320, 90.247, 96.281])

fig, ax = plt.subplots(figsize=(10.0, 6.2), facecolor="white")
ax.set_facecolor("white")

b1 = ax.bar(x - 1.5*w, random_peak, width=w, color=BLUE, edgecolor="white", linewidth=1.0,
            label="Peak recall · Random", zorder=3)
b2 = ax.bar(x - 0.5*w, scaffold_peak, width=w, color=BLUE2, edgecolor="white", linewidth=1.0,
            label="Peak recall · Scaffold", zorder=3)
b3 = ax.bar(x + 0.5*w, random_int, width=w, color=PURPLE, edgecolor="white", linewidth=1.0,
            label="Explained intensity · Random", zorder=3)
b4 = ax.bar(x + 1.5*w, scaffold_int, width=w, color=PURPLE2, edgecolor="white", linewidth=1.0,
            label="Explained intensity · Scaffold", zorder=3)

def annotate(bars, dy=1.0, fs=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + dy,
            f"{h:.1f}",
            ha="center", va="bottom",
            fontsize=fs, color=INK
        )

annotate(b1)
annotate(b2)
annotate(b3)
annotate(b4)

ax.set_title("A  Candidate-space coverage", loc="left", color=INK, fontsize=20, fontweight="bold", pad=10)
ax.set_ylabel("Coverage (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 103)

ax.grid(axis="y", color=GRID, linewidth=1.0, alpha=0.9, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.4)
ax.spines["bottom"].set_linewidth(1.4)
ax.tick_params(axis="both", width=1.4, length=6, colors=INK, pad=6)

ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.4)

fig.tight_layout()
fig.savefig(OUT / "Fig2A_candidate_coverage.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig2A_candidate_coverage.png", dpi=600, bbox_inches="tight")
print(OUT / "Fig2A_candidate_coverage.pdf")
print(OUT / "Fig2A_candidate_coverage.png")

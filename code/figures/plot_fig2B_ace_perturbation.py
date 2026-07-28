from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3B82C4"
PURPLE = "#76639A"
ORANGE = "#D8A24B"
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

metrics = ["CBIN", "JSS", "CHUN"]
x = np.arange(len(metrics))
w = 0.22

true = np.array([0.656, 0.622, 0.620])
median = np.array([0.504, 0.499, 0.475])
shuffled = np.array([0.448, 0.454, 0.424])

true_sd = np.array([0.006, 0.005, 0.005])
median_sd = np.array([0.006, 0.004, 0.006])
shuffled_sd = np.array([0.003, 0.003, 0.003])

fig, ax = plt.subplots(figsize=(10.0, 6.2), facecolor="white")
ax.set_facecolor("white")

b1 = ax.bar(x - w, true, width=w, yerr=true_sd, capsize=4,
            color=BLUE, edgecolor="white", linewidth=1.0,
            label="True ACE", zorder=3)
b2 = ax.bar(x, median, width=w, yerr=median_sd, capsize=4,
            color=PURPLE, edgecolor="white", linewidth=1.0,
            label="Median ACE", zorder=3)
b3 = ax.bar(x + w, shuffled, width=w, yerr=shuffled_sd, capsize=4,
            color=ORANGE, edgecolor="white", linewidth=1.0,
            label="Shuffled ACE", zorder=3)

def annotate(bars, fs=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.012,
            f"{h:.3f}",
            ha="center", va="bottom",
            fontsize=fs, color=INK
        )

annotate(b1)
annotate(b2)
annotate(b3)

ax.set_title("B  Collision-energy perturbation", loc="left", color=INK, pad=14)
ax.set_ylabel("Similarity")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight="bold")
ax.set_ylim(0, 0.72)

ax.grid(axis="y", color=GRID, linewidth=1.0, alpha=0.9, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.4)
ax.spines["bottom"].set_linewidth(1.4)
ax.tick_params(axis="both", width=1.4, length=6, colors=INK, pad=6)

ax.legend(frameon=False, loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.2)

fig.tight_layout()
fig.savefig(OUT / "Fig2B_ace_perturbation.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig2B_ace_perturbation.png", dpi=600, bbox_inches="tight")
print(OUT / "Fig2B_ace_perturbation.pdf")
print(OUT / "Fig2B_ace_perturbation.png")

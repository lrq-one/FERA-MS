from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#123B7A"
TEAL = "#6FA9A3"
PURPLE = "#76639A"
ORANGE = "#D8A24B"
GRAY = "#93A0AF"
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
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

groups = ["Low\n≤20 eV", "Middle\n20–40 eV", "High\n>40 eV"]
x = np.arange(len(groups))
w = 0.14

series = [
    ("FERA-MS", NAVY, np.array([0.694, 0.619, 0.530])),
    ("NEIMS-ACE", TEAL, np.array([0.564, 0.521, 0.424])),
    ("MassFormer-ACE", PURPLE, np.array([0.515, 0.470, 0.340])),
    ("FraGNNet-D3-ACE", ORANGE, np.array([0.461, 0.332, 0.250])),
    ("ICEBERG-core", GRAY, np.array([0.402, 0.313, 0.187])),
]

fig, ax = plt.subplots(figsize=(10.0, 6.3), facecolor="white")
ax.set_facecolor("white")

offsets = np.array([-2, -1, 0, 1, 2]) * w
all_bars = []

for (name, color, vals), off in zip(series, offsets):
    bars = ax.bar(
        x + off, vals, width=w, color=color,
        edgecolor="white", linewidth=1.0,
        label=name, zorder=3
    )
    all_bars.append((name, bars))

for name, bars in all_bars:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.008,
            f"{h:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, color=INK, rotation=0
        )

ax.text(
    0.02, 0.98, "Best at every ACE stratum",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=14, color=NAVY, fontweight="bold"
)

ax.set_title("C  ACE-stratified spectrum prediction", loc="left", color=INK, pad=14)
ax.set_ylabel("Micro CBIN")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylim(0.15, 0.76)

ax.grid(axis="y", color=GRID, linewidth=1.0, alpha=0.9, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.4)
ax.spines["bottom"].set_linewidth(1.4)
ax.tick_params(axis="both", width=1.4, length=6, colors=INK, pad=6)

ax.legend(
    frameon=False,
    loc="upper right",
    ncol=2,
    columnspacing=0.8,
    handlelength=1.1,
    handletextpad=0.4,
    borderaxespad=0.3
)

fig.tight_layout()
fig.savefig(OUT / "Fig2C_ace_strata.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig2C_ace_strata.png", dpi=600, bbox_inches="tight")
print(OUT / "Fig2C_ace_strata.pdf")
print(OUT / "Fig2C_ace_strata.png")

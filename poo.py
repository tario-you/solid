import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------------------------------------------------------
# EXAMPLE DATA: 36 rows (3 categories × 12 months), 10 columns (tickers)
# Replace df_pnl_total with your actual DataFrame of PnL.
# ----------------------------------------------------------------------------

# 1) Build a colormap mapping:
#      -200  → fraction=0.0   → red
#       0    → fraction=0.1667→ (1,1,0.8)
#     1000   → fraction=1.0   → green
# The fraction for 0 is (0 - (-200)) / (1000 - (-200)) = 200 / 1200 ≈ 0.1667
my_cmap = mcolors.LinearSegmentedColormap.from_list(
    'ManualCmap',
    [
        (0.0,    (1, 0, 0)),     # red
        (0.1667, (1, 1, 0.8)),   # light yellow
        (1.0,    (0, 1, 0)),     # green
    ],
    N=256
)

# 2) Simple linear Normalize from -200 to 1000, clipping out-of-range data
my_norm = mcolors.Normalize(vmin=-200, vmax=1000, clip=True)

# ----------------------------------------------------------------------------
# 3) Plot the heatmap
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    df_pnl_total,
    cmap=my_cmap,
    norm=my_norm,
    annot=False,
    square=True,
    linewidths=0.5,
    linecolor="black",
    ax=ax
)

# 4) Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=my_norm)
sm.set_array([])  # Dummy array

# 5) Attach colorbar to the same figure/axes as the heatmap
#    'extend="neither"' ensures the bar starts exactly at red (–200)
#    and ends at green (1000), with no extra arrows or extension.
cbar = ax.figure.colorbar(
    sm,
    ax=ax,
    ticks=[-200, 0, 1000],
    extend='neither',     # no extension beyond our min/max
    fraction=0.045,
    pad=0.03
)
cbar.set_label("PnL", fontsize=12)

# Optional: Tidy up
ax.set_title("PnL Heatmap: -200 = Red, 0 = Light Yellow, 1000 = Green", pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

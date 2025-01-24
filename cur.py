plt.figure(figsize=(20, 12))

min_val = pnl_coordinated.values.min()
max_val = pnl_coordinated.values.max()

# 1) Get the built-in RdYlGn colormap
base_cmap = plt.get_cmap("RdYlGn", 256)  # 256 discrete colors

# 2) Convert it to a list so we can modify the middle band
colors = [base_cmap(i) for i in range(base_cmap.N)]

# 3) Make the midpoint less yellow. For example:
#    - The midpoint in a 256-color map is index ~128
#    - Replace it with something lighter (blend with white).
mid_index = 128
# RGBA of the original midpoint (~ bright yellow)
original_mid = colors[mid_index]
# Let's blend that original color with white at, say, 70% original / 30% white:
blend_ratio = 0.7
new_mid = (
    original_mid[0] * blend_ratio + 1.0 * (1 - blend_ratio),
    original_mid[1] * blend_ratio + 1.0 * (1 - blend_ratio),
    original_mid[2] * blend_ratio + 1.0 * (1 - blend_ratio),
    1.0  # keep alpha=1
)
colors[mid_index] = new_mid

# You can also adjust a small band around the midpoint if you want a wider, paler zone
# For example, re-blend indices [120..135] to smoothen the transition:
for idx in range(120, 136):
    c = colors[idx]
    colors[idx] = (
        c[0] * blend_ratio + 1.0 * (1 - blend_ratio),
        c[1] * blend_ratio + 1.0 * (1 - blend_ratio),
        c[2] * blend_ratio + 1.0 * (1 - blend_ratio),
        1.0
    )

# 4) Create a new colormap from our modified colors
my_cmap = mcolors.LinearSegmentedColormap.from_list(
    "RedYellowGreen",
    [
        (1, 0, 0),  # Red   = rgb(255,0,0)
        (1, 1, 0.8),  # Yellow= rgb(255,255,0)
        (0, 1, 0)   # Green = rgb(0,255,0)
    ],
    N=256
)

my_norm = FullSaturationNorm(neg_bound=-200, pos_bound=1000)

ax = sns.heatmap(
    df_pnl_total,
    cmap=my_cmap,
    norm=my_norm,
    # center=0.0,       # midpoint at zero
    annot=False,
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor="black"
)

# sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=my_norm)
# sm.set_array([])  # dummy array for colorbar
# cbar = plt.colorbar(sm, ticks=[-200, 0, 1000], fraction=0.045, pad=0.03)
# cbar.set_label("PnL", fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

ax.tick_params(axis="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tick_params(axis="x", top=True, labeltop=True, labelbottom=False)
plt.xlabel(None)

n_rows = df_pnl_total.shape[0]  # should be 36
# We'll label each row with i % 12 => repeats 0..11 three times
ytick_positions = np.arange(n_rows) + 0.5  # center labels in each row
ytick_labels = [str(i % 12) for i in range(n_rows)]

ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels, rotation=0)

plt.ylabel("Month\n\n\n\n")

# Vertical/horizontal offsets for the bracket
y_top = 1.075
y_bottom = y_top-0.01
margin = 0.3  # how much to pull in from each side so brackets don't overlap
linewidth = 0.75

for i, cat in enumerate(categories):
    x_left = i * 6 + margin
    x_right = (i + 1) * 6 - margin

    # Left vertical line
    ax.plot([x_left, x_left], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Right vertical line
    ax.plot([x_right, x_right], [y_bottom, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Horizontal top line
    ax.plot([x_left, x_right], [y_top, y_top],
            color="black", lw=linewidth, transform=ax.get_xaxis_transform(), clip_on=False)
    # Category label
    ax.text((x_left + x_right) / 2, y_top + 0.02, '\n'.join(cat.split())+'',
            ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10)

# We'll place them just beyond the last column:
bracket_spans = [(0, 12), (12, 24), (24, 36)]
labels = ["coordinator", "llm", "opt"]
bracket_x_right = -1.5
bracket_x_left = bracket_x_right + 0.4
linewidth = 0.75

for (y_bottom, y_top), cat_label in zip(bracket_spans, labels):
    # Bottom horizontal line
    ax.plot([bracket_x_left, bracket_x_right], [y_bottom, y_bottom],
            color="black", lw=linewidth, clip_on=False)
    # Top horizontal line
    ax.plot([bracket_x_left, bracket_x_right], [y_top, y_top],
            color="black", lw=linewidth, clip_on=False)
    # Vertical line
    ax.plot([bracket_x_right, bracket_x_right], [y_bottom, y_top],
            color="black", lw=linewidth, clip_on=False)

    # Text label in middle
    y_mid = (y_bottom + y_top) / 2
    x_text = bracket_x_right-0.4
    ax.text(x_text, y_mid, cat_label, va="center", ha="right",
            fontsize=12, clip_on=False)

# After creating ax = sns.heatmap(...):
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=my_norm)
sm.set_array([])  # dummy array

# Instead of plt.colorbar(...), do:
cbar = ax.figure.colorbar(
    sm, ax=ax,
    ticks=[-200, 0, 1000],
    fraction=0.045,
    extend='neither',
    pad=0.03
)
cbar.set_label("PnL", fontsize=12)

plt.title("Monthly PnL per Ticker", fontsize=16, pad=85)
plt.tight_layout()

plt.show()

plt.figure(figsize=(20, 12))

min_val = df_pnl_total.values.min()
max_val = df_pnl_total.values.max()

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
    'ManualCmap',
    [
        (0.0,    (1, 0, 0)),     # red
        (0.1667, (1, 1, 0.8)),   # light yellow
        (1.0,    (0, 1, 0)),     # green
    ],
    N=256
)

my_norm = mcolors.Normalize(vmin=-200, vmax=1000, clip=True)


# heatmap:
mappable = None

for i in range(5):
    ax = axes[i]

    sub_df = df_pnl_total.iloc[i*12: (i+1)*12, :]

    # Create the heatmap with no colorbar
    hmap = sns.heatmap(
        sub_df,
        cmap=my_cmap,
        norm=my_norm,
        annot=False,
        cbar=False,      # No inline colorbar
        square=True,
        linewidths=0.5,
        linecolor="black"
    )

    # Save the "mappable" from the last heatmap in the loop.
    # We can use any subplot's "mappable" for the colorbar,
    # but just store one (e.g. from the last iteration).
    mappable = hmap.collections[0]

    # Turn off bottom tickers; optionally place them on top.
    ax.tick_params(axis='x',
                   bottom=False, labelbottom=False,   # Turn off bottom
                   top=True, labeltop=True,           # Put ticks on top
                   length=0)

    # y-axis labels (just an example: row numbers 0..11)
    ytick_positions = np.arange(sub_df.shape[0]) + 0.5
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([str(y) for y in range(sub_df.shape[0])], rotation=0)

    # Label each subplot on the y-axis with your desired text
    ax.set_ylabel(labels[i], fontsize=12)

    # Show only bottom & right spines; hide top & left
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Rotate x-tick labels on top
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# -- A SINGLE COLORBAR ON THE RIGHT, SPANNING ALL SUBPLOTS --
# 'mappable' comes from the last heatmap above.
cbar = fig.colorbar(
    mappable,  # ???
    ax=axes.ravel().tolist(),   # attach to all subplots
    orientation='vertical',
    fraction=0.02,
    pad=0.03
)

# Vertical/horizontal offsets for the bracket
y_top = 6.18
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
    ax.text((x_left + x_right) / 2, y_top + 0.01, '\n'.join(cat.split())+'',
            ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10)

# -- ADD AN OVERALL TITLE --
fig.suptitle("Monthly PnL per Ticker", fontsize=20, y=0.94)

plt.savefig(pnl_path, dpi=500, bbox_inches='tight')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

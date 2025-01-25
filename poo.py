# Example data: Suppose df_total has 60 rows (5 sets of 12).
# Adapt to match your real data shape.
n_rows = 60
n_cols = 10

# -- FIGURE AND AXES --
fig, axes = plt.subplots(nrows=5, figsize=(20, 23), sharex=True)

# -- SETUP LOG NORM --
vmin = 1e-3
vmax = df_total.values.max()
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# We'll create a single colorbar at the end, so set cbar=False for each subplot
# The data range is the same for all subplots, so we can just pick the last heatmap's
# "mappable" to feed into fig.colorbar() later.
mappable = None

labels = ["LLM+OPT", "LLM_sparse+OPT", "LLM", "LLM_sparse", "OPT"]

for i in range(5):
    ax = axes[i]

    sub_df = df_total.iloc[i*12: (i+1)*12, :]

    # Create the heatmap with no colorbar
    hmap = sns.heatmap(
        sub_df,
        cmap="Reds",
        linewidths=0.5,
        linecolor="black",
        annot=False,
        square=True,
        cbar=False,      # No inline colorbar
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        ax=ax
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
    mappable,
    ax=axes.ravel().tolist(),   # attach to all subplots
    orientation='vertical',
    fraction=0.02,
    pad=0.03
)
# Adjust the ticks & labels on the colorbar
cbar.set_ticks([1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])

categories = [
    'Technology',
    'Consumer Discretionary',
    'Financials',
    'Real Estate',
    'Energy',
    'Healthcare',
    'Industrials',
    'Materials',
    'Communication Services',
    'Consumer Staples'
]

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
fig.suptitle("Weights per Stock", fontsize=20, y=0.94)

plt.savefig(heatmap_all_path, dpi=500, bbox_inches='tight')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

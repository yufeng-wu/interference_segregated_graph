# Semi-synthetic data plot

import matplotlib.pyplot as plt
import pandas as pd

# Networks and Tests
NETWORK_NAMES = ["HR_edges", "HU_edges", "RO_edges", "deezer_europe_edges", "lastfm_asia_edges"]
TEST_NAMES = ["L", "A", "Y"]

# Colors for each network
color_map = {
    "HR_edges": "blue",
    "HU_edges": "green",
    "RO_edges": "purple",
    "deezer_europe_edges": "orange",
    "lastfm_asia_edges": "red"
}

# Line styles for metrics
linestyle_map = {
    "power": "-",
    "type_I_error_rate": "--"
}

# Create figure with 3 subplots in a row
# Increase the height from 6 to 8 (or more) for extra vertical space
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), sharey=True)

# Overarching title
fig.suptitle("Performance Across L, A, and Y Layers", fontsize=14, y=1.02)

for i, test_name in enumerate(TEST_NAMES):
    ax = axes[i]
    for network in NETWORK_NAMES:
        df = pd.read_csv(f"./result/{network}/{network}_{test_name}_results_layer_only.csv")

        # Plot Power
        ax.plot(
            df["n_units"],
            df["power"],
            color=color_map[network],
            linestyle=linestyle_map["power"],
            linewidth=2,
            label=f"{network} Power"
        )

        # Plot Type I Error
        ax.plot(
            df["n_units"],
            df["type_I_error_rate"],
            color=color_map[network],
            linestyle=linestyle_map["type_I_error_rate"],
            linewidth=2,
            label=f"{network} Type I Error"
        )

    # Horizontal significance line
    ax.axhline(
        y=0.05,
        color="red",
        linestyle=":",
        linewidth=2,
        label="Significance Level (0.05)"
    )

    ax.set_title(f"{test_name} Layer", fontsize=14)
    ax.set_xlabel("Sample Size", fontsize=12)
    if i == 0:
        ax.set_ylabel("Rate", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

# Create a combined legend
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))

fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),  # Shift legend lower if needed
    ncol=4,
    fontsize=12
)

# Leave space for title at top and legend at bottom
plt.tight_layout(rect=[0, 0.08, 1, 0.93])

# Save with extra padding to avoid clipping
plt.savefig(
    "./result/plot/LAY_comparison_semi_synthetic.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.5
)

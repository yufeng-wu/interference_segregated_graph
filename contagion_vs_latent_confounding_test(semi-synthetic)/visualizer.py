import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
NETWORK_NAMES = ["HR_edges", "HU_edges", "RO_edges", "deezer_europe_edges", "lastfm_asia_edges"]

# Color map for the networks
color_map = {
    "HR_edges": "blue",
    "HU_edges": "green",
    "RO_edges": "purple",
    "deezer_europe_edges": "orange",
    "lastfm_asia_edges": "red"
}

# Paths (adjust if needed)
BASE_PATH = "/Users/dingding/Desktop/Thesis/code/contagion_vs_latent_confounding_test(semi-synthetic)"
RESULTS_FOLDER = "result"
INTERMEDIATE_FOLDER = "intermediate_data"

# We'll store all needed info in a dictionary:
# data_dict[network] = {
#     "burnin": <int>,
#     "average_ess": <float>,
#     "results": {
#          "L": {"type_I_error": <float>, "power": <float>},
#          "A": {"type_I_error": <float>, "power": <float>},
#          "Y": {"type_I_error": <float>, "power": <float>}
#      }
# }
data_dict = {}

# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------
for net in NETWORK_NAMES:
    # 1) Read the result CSV
    result_csv_path = os.path.join(BASE_PATH, RESULTS_FOLDER, net, f"{net}_result.csv")
    df_result = pd.read_csv(result_csv_path)
    
    outcome_dict = {}
    for _, row in df_result.iterrows():
        outcome_dict[row["outcome"]] = {
            "type_I_error": row["type_I_error"],
            "power": row["power"]
        }

    # 2) Read the log file to extract average ESS
    log_txt_path = os.path.join(BASE_PATH, RESULTS_FOLDER, net, f"{net}_log.txt")
    average_ess = None
    with open(log_txt_path, "r") as f:
        for line in f:
            if "Average effective sample size" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    average_ess = float(parts[1].strip())
                break

    # 3) Read the burn-in period
    burn_in_path = os.path.join(BASE_PATH, INTERMEDIATE_FOLDER, net, f"{net}_burn_in.txt")
    burn_in = None
    with open(burn_in_path, "r") as f:
        for line in f:
            if "Burn-in period" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    burn_in = int(parts[1].strip())
                break
    
    data_dict[net] = {
        "burnin": burn_in,
        "average_ess": average_ess,
        "results": outcome_dict
    }

# --------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

outcomes = ["L", "A", "Y"]
markers = {"type_I_error": "o", "power": "x"}  # Different markers

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    ax.set_title(f"{outcome} Layer")  # e.g. "L Layer"
    ax.set_xlabel("Average effective sample size")
    
    if i == 0:
        ax.set_ylabel("Type I error rate / Power")
    
    # Draw the red dashed line for significance level
    ax.axhline(y=0.05, color='red', linestyle='--', label="Significance level (α=0.05)")
    
    # Plot each network’s data
    for net in NETWORK_NAMES:
        net_data = data_dict[net]
        avg_ess  = net_data["average_ess"]
        y_type_i = net_data["results"][outcome]["type_I_error"]
        y_power  = net_data["results"][outcome]["power"]
        
        # Scatter for Type I error
        ax.scatter(
            avg_ess, 
            y_type_i, 
            color=color_map[net],
            marker=markers["type_I_error"],
            s=60
        )
        # Scatter for Power
        ax.scatter(
            avg_ess, 
            y_power, 
            color=color_map[net],
            marker=markers["power"],
            s=60
        )

# --------------------------------------------------------------------
# Build the legend(s)
# --------------------------------------------------------------------

# Sort networks by ascending ESS
sorted_networks = sorted(NETWORK_NAMES, key=lambda net: data_dict[net]['average_ess'])

custom_labels = {
    "lastfm_asia_edges": "LastFM Asia",
    "HR_edges": "HR",
    "deezer_europe_edges": "Deezer Europe",
    "HU_edges": "HU",
    "RO_edges": "RO"
}

# --------------------------------------------------------------------
# 1) Network legend (row 1) - no lines, just markers, sorted by ESS
# --------------------------------------------------------------------
network_legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color=color_map[net],  # Use the network color directly, no edge
        label=f"{custom_labels[net]} (burn-in={data_dict[net]['burnin']})",  # Use the custom labels
        markersize=8,
        linestyle='None'  # No line
    )
    for net in sorted_networks
]

# --------------------------------------------------------------------
# 2) Markers legend (row 2) - 'o' for Type I error, 'x' for Power
# --------------------------------------------------------------------
marker_legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='black',  # Solid black circle
        label='Type I error rate',
        markersize=8,
        linestyle='None'  # No line
    ),
    Line2D(
        [0], [0],
        marker='x',
        color='black',  # Solid black cross
        label='Power',
        markersize=8,
        linestyle='None'  # No line
    )
]

# --------------------------------------------------------------------
# Add Legends Separately and Adjust Positioning
# --------------------------------------------------------------------

# Create the first row of the legend (centered): Network labels
legend_networks = fig.legend(
    handles=network_legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.15),  # Adjust y position for spacing
    ncol=len(network_legend_elements),  # Networks side-by-side
    title="Networks",
    frameon=False
)

# Create the second row of the legend (centered): Marker types
legend_markers = fig.legend(
    handles=marker_legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),  # Position below the network legend
    ncol=len(marker_legend_elements),  # Markers side-by-side
    title="Markers",
    frameon=False
)

# Adjust the title position to add more space between the title and plots
fig.suptitle(
    "Performance of likelihood ratio test on semi-synthetic data",
    fontsize=14
)

# Ensure there's enough space for the legends at the bottom
plt.subplots_adjust(bottom=0.4)

# Finally, save the figure
plt.savefig("./result/plot/semi_synthetic_result.png")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

filename = "BUU"

fontsize = 14

# Read data from CSV files
df_ours = pd.read_csv(f'../result/raw_output/{filename}_ours.csv')
df_autog = pd.read_csv(f'../result/raw_output/{filename}_autog.csv')

# Define colors for the boxplots
ours_color = "skyblue"
autog_color = "lightgreen"

# Get the sample sizes from the column names
n_samples = [int(column_name.split()[2]) for column_name in df_ours.columns if 'n units' in column_name]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Set titles for subplots
axs[0].set_title(f'Our Method Estimates {filename}', fontsize=fontsize)
axs[1].set_title(f'Auto-G Estimates {filename}', fontsize=fontsize)

for ax, df, color, label in zip(axs, [df_ours, df_autog], [ours_color, autog_color], ['Our Method', 'Auto-G']):
    for n_sample in n_samples:
        ax.boxplot(df[f'n units {n_sample}'], positions=[n_sample], widths=200,
               showfliers=True, patch_artist=True,
               boxprops=dict(facecolor=color))
        ax.set_xlabel('Number of units', fontsize=fontsize)
    
    # Add a line for the true causal effect for each subplot
    true_causal_effect = (df_ours['True Effect'][0] + df_autog['True Effect'][0]) / 2
    ax.axhline(y=true_causal_effect, color='r', linestyle='--')
    # Set font size for ticks
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # Create legend with a rectangle patch for color
    handles = [plt.Rectangle((0, 0), 1, 1, color=color), plt.Line2D([0], [0], color='r', linestyle='--')]
    ax.legend(handles, [label, 'True Causal Effect'], loc='best', fontsize=fontsize)

# Determine the overall minimum and maximum y-values from the datasets
min_y_value = min(df_ours.min().min(), df_autog.min().min())
max_y_value = max(df_ours.max().max(), df_autog.max().max())

# Set some padding for the min and max y-values
padding = (max_y_value - min_y_value) * 0.1  # 10% padding
min_ylim = min_y_value - padding
max_ylim = max_y_value + padding

# Use the same y-axis limits for both subplots
axs[0].set_ylim(min_ylim, max_ylim)
axs[1].set_ylim(min_ylim, max_ylim)

# Set font size for all texts in the plots
plt.rcParams.update({'font.size': fontsize})

plt.tight_layout()
# Save the figure
plt.savefig(f'../result/plot/{filename}.png')

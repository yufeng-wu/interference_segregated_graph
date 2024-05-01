# This code visualizes results from likelihood_ratio_test.py.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in data
data_l = pd.read_csv('./result/L_results.csv')
data_a = pd.read_csv('./result/A_results.csv')
data_y = pd.read_csv('./result/Y_results.csv')

# Convert dictionaries to DataFrames
df_l = pd.DataFrame(data_l)
df_a = pd.DataFrame(data_a)
df_y = pd.DataFrame(data_y)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), gridspec_kw={'wspace': 0.3})
fontsize = 14

# Plot for L results
axes[0].plot(df_l['n_units'], df_l['type_I_error_rate'], label='Type I Error Rate', marker='o', color='green')
axes[0].plot(df_l['n_units'], df_l['power'], label='Power', marker='o', color='blue')
axes[0].axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α = 0.05)')
axes[0].set_title('L Layer Test Performance', fontsize=fontsize)
axes[0].set_xlabel('Sample Size', fontsize=fontsize)
axes[0].set_ylabel('Rate', fontsize=fontsize)

# Plot for A results
axes[1].plot(df_a['n_units'], df_a['type_I_error_rate'], label='Type I Error Rate', marker='o', color='green')
axes[1].plot(df_a['n_units'], df_a['power'], label='Power', marker='o', color='blue')
axes[1].axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α = 0.05)')
axes[1].set_title('A Layer Test Performance', fontsize=fontsize)
axes[1].set_xlabel('Sample Size', fontsize=fontsize)
axes[1].set_ylabel('Rate', fontsize=fontsize)

# Plot for Y results
axes[2].plot(df_y['n_units'], df_y['power'], label='Power', marker='o', color='blue')
axes[2].plot(df_y['n_units'], df_y['type_I_error_rate'], label='Type I Error Rate', marker='o', color='green')
axes[2].axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α = 0.05)')
axes[2].set_title('Y Layer Test Performance', fontsize=fontsize)
axes[2].set_xlabel('Sample Size', fontsize=fontsize)
axes[2].set_ylabel('Rate', fontsize=fontsize)

axes[2].legend(fontsize=fontsize, loc='center', bbox_to_anchor=(0.8, 0.5))

plt.savefig('./result/plot/combined_test_result.png')
# Plot three tests in a single plot (for synthetic data)

import matplotlib.pyplot as plt
import pandas as pd

# Read in data, modify the file names as needed.
data_l = pd.read_csv('./result/L_results_layer_only.csv')
data_a = pd.read_csv('./result/A_results_layer_only.csv')
data_y = pd.read_csv('./result/Y_results_layer_only.csv')

# Convert dictionaries to DataFrames
df_l = pd.DataFrame(data_l)
df_a = pd.DataFrame(data_a)
df_y = pd.DataFrame(data_y)

fig, ax = plt.subplots(figsize=(10, 6))
fontsize = 15

# Plot power for each test
ax.plot(df_l['n_units'], df_l['power'], label='L Layer Test Power', marker='o', color='blue')
ax.plot(df_a['n_units'], df_a['power'], label='A Layer Test Power', marker='o', color='green')
ax.plot(df_y['n_units'], df_y['power'], label='Y Layer Test Power', marker='o', color='orange')

# Plot Type I error rate for each test
ax.plot(df_l['n_units'], df_l['type_I_error_rate'], label='L Layer Test Type I Error Rate', marker='^', linestyle='-', color='blue')
ax.plot(df_a['n_units'], df_a['type_I_error_rate'], label='A Layer Test Type I Error Rate', marker='^', linestyle='-', color='green')
ax.plot(df_y['n_units'], df_y['type_I_error_rate'], label='Y Layer Test Type I Error Rate', marker='^', linestyle='-', color='orange')

# Plot significance level line
ax.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α = 0.05)')

# Set labels and title
ax.set_title('Test Performance', fontsize=fontsize)
ax.set_xlabel('Sample Size', fontsize=fontsize)
ax.set_ylabel('Rate', fontsize=fontsize)

# Adjust legend
ax.legend(fontsize=12, loc='best')

ax.tick_params(axis='both', which='major', labelsize=16)

# Save output, modify the file names as needed.
plt.savefig('./result/plot/combined_test_result_layer_only_all_in_one.png')
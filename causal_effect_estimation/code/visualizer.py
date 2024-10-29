import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# edit this to the filename of the data you want to visualize
filename = "UBU_autog" 

fontsize = 14 # font size of the axis and title

df = pd.read_csv(f'../result/raw_output/{filename}.csv')
n_samples = [int(column_name.split()[2]) for column_name in df.columns if 'n units' in column_name]
name, _ = filename.split('_')

plt.figure(figsize=(10, 6))
for n_sample in n_samples:
    # Option to add some jitter and plot the data points
    x_vals = np.random.normal(n_sample, 100, len(df[f'n units {n_sample}']))
    # plt.plot(x_vals, df[f'n units {n_sample}'], 'ro', markersize=2, alpha=0.3)
    
    plt.boxplot(df[f'n units {n_sample}'], positions=[n_sample], widths=200, 
            showfliers=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))


# add a line for the true causal effect
true_effect = df['True Effect'][0]
plt.axhline(y=true_effect, color='r', linestyle='--')

plt.xticks(n_samples, fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of units', fontsize=fontsize)
plt.ylabel('Causal Effect', fontsize=fontsize)
plt.title(f'Causal Effect Estimates Using Auto-G: {name}', fontsize=fontsize)
# plt.show()

plt.savefig(f'../result/plot/{filename}.png')

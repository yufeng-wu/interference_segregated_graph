import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "BBB_ours"
dirname = ""

df = pd.read_csv(f'./result{dirname}/{filename}.csv')
n_samples = [int(column_name.split()[2]) for column_name in df.columns if 'n units' in column_name]

plt.figure(figsize=(10, 6))
for n_sample in n_samples:
    # add some jitter
    x_vals = np.random.normal(n_sample, 10, len(df[f'n units {n_sample}']))
    plt.plot(x_vals, df[f'n units {n_sample}'], 'ro', markersize=2)
    plt.boxplot(df[f'n units {n_sample}'], positions=[n_sample], widths=200)

# also add a line for the true causal effect
plt.axhline(y=df['True Effect'][0], color='r', linestyle='--')

plt.xlabel('Number of units')
plt.ylabel('Causal Effect')
plt.title(f'Causal Effect Estimates {filename}')
# plt.show()

plt.savefig(f'./result{dirname}/{filename}.png')

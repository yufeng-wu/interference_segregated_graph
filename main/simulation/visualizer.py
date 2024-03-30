import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./result/2000_UUB_ours.csv')
n_samples = [int(column_name.split()[2]) for column_name in df.columns if 'n units' in column_name]

plt.figure(figsize=(10, 6))
for n_sample in n_samples:
    # add some jitter
    x_vals = np.random.normal(n_sample, 100, len(df[f'n units {n_sample}']))
    plt.plot(x_vals, df[f'n units {n_sample}'], 'ro', markersize=2)
    plt.boxplot(df[f'n units {n_sample}'], positions=[n_sample], widths=50, sym="")

# also add a line for the true causal effect
plt.axhline(y=df['True Effect'][0], color='r', linestyle='--')
plt.xlim(min(n_samples) -100, max(n_samples) + 100)

plt.xlabel('Number of units')
plt.ylabel('Causal Effect')
plt.title('Causal Effect Estimates for UUB Data')
plt.show()

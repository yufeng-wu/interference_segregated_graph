import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./result/UUB_ours.csv')
n_samples = [int(column_name.split()[2]) for column_name in df.columns if 'n units' in column_name]

plt.figure(figsize=(10, 6))
for n_sample in n_samples:
    plt.boxplot(df[f'n units {n_sample}'], positions=[n_sample], widths=50)
    
# also add a line for the true causal effect
plt.axhline(y=df['True Effect'][0], color='r', linestyle='--')

plt.xlabel('Number of units')
plt.ylabel('Causal Effect')
plt.title('Causal Effect Estimates for UUB Data')
plt.show()




import pandas as pd

# Load the data from CSV files
df1 = pd.read_csv('../result/raw_output/UBB_ours_5000.csv')
df2 = pd.read_csv('../result/raw_output/UBB_ours_6000.csv')

# Temporarily remove 'True Effect' from df1 and store it
true_effect = df1['True Effect']
df1 = df1.drop(columns=['True Effect'])

# Drop the 'True Effect' column from the second DataFrame
df2 = df2.drop(columns=['True Effect'])

# Join the DataFrames horizontally
result_df = pd.concat([df1, df2], axis=1)

# Append 'True Effect' at the end
result_df['True Effect'] = true_effect

# Save the combined DataFrame back to a CSV file if needed
result_df.to_csv('../result/raw_output/combined_dataset.csv', index=False)

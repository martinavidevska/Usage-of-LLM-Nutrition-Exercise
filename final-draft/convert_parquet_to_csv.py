import pandas as pd

# Load both datasets
cleaned_df = pd.read_csv('cleaned_dataset.csv')
combined_df = pd.read_csv('combined_dataset.csv')

# Check the structure of both datasets to ensure they have compatible columns
print(f"cleaned_dataset columns: {cleaned_df.columns.tolist()}")
print(f"combined_dataset columns: {combined_df.columns.tolist()}")

# Remove the 'source' column from combined_df if it exists
if 'source' in combined_df.columns:
    combined_df = combined_df.drop(columns=['source'])

# Rename columns in both dataframes to "Query" and "Result"
cleaned_df.columns = ['Query', 'Result']
combined_df.columns = ['Query', 'Result']

# Combine the datasets
merged_df = pd.concat([cleaned_df, combined_df], ignore_index=True)

# Optionally, drop any duplicate rows
merged_df = merged_df.drop_duplicates()

# Save the combined dataset
merged_df.to_csv('full_merged_dataset.csv', index=False)

print(f"Successfully combined datasets with {len(merged_df)} total rows")
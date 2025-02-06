import pandas as pd

# Load the dataset
df_original = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Create a copy of the dataset for processing
df = df_original.copy()

# Display the first 5 rows
print("Original Dataset (First 5 Rows):")
print(df.head())

# Display dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values in Original Dataset:")
print(df.isnull().sum())

# Check for empty strings in TotalCharges
empty_strings = df['TotalCharges'].eq(' ')
print(f"\nNumber of empty strings in TotalCharges: {empty_strings.sum()}")

# Convert TotalCharges to numeric (empty strings will become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill NaN values with 0
df['TotalCharges'].fillna(0, inplace=True)

# Verify no missing values remain
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save the cleaned dataset to a new file
df.to_csv('../data/telco_churn_cleaned.csv', index=False)
print("\nCleaned dataset saved to '../data/telco_churn_cleaned.csv'.")
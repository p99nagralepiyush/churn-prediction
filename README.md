## Data Cleaning Steps
1. Checked for missing values (`NaN`) in the original dataset.
2. Identified 11 empty strings in the `TotalCharges` column.
3. Converted `TotalCharges` to numeric, replacing empty strings with `NaN`.
4. Filled `NaN` values in `TotalCharges` with `0`.
5. Saved the cleaned dataset to `data/telco_churn_cleaned.csv`.

## Key Observations
- No missing values (`NaN`) in the original dataset.
- `TotalCharges` contained 11 empty strings, which were replaced with `0`.
- Categorical columns need encoding (e.g., `gender`, `Partner`).
- Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) need scaling.
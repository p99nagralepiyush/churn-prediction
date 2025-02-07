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

## Data Preprocessing Steps
1. **Encoded Categorical Variables**:  
   - Used `LabelEncoder` to convert categorical columns (e.g., `gender`, `Partner`) into numerical format.  
2. **Scaled Numerical Features**:  
   - Used `StandardScaler` to normalize numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`).  
3. **Split Data into Train/Test Sets**:  
   - Split the dataset into training (80%) and testing (20%) sets.  
4. **Saved Processed Data**:  
   - Saved the encoded, scaled, and split datasets to the `data/` folder.  

   ## Baseline Model Results
- **Algorithm**: Logistic Regression
- **Performance Metrics**:
  - Accuracy: 0.8169
  - Precision: 0.6803
  - Recall: 0.5818
  - F1-Score: 0.6272
  - AUC-ROC: 0.8615
- **Next Steps**:
  - Address class imbalance (e.g., using SMOTE).
  - Experiment with more advanced models (e.g., Random Forest, XGBoost).
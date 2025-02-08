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

  ## Advanced Modeling Results (XGBoost)
- **Algorithm**: XGBoost
- **Performance Metrics (Before Tuning)**:
  - Accuracy: 0.7956
  - Precision: 0.6358
  - Recall: 0.5335
  - F1-Score: 0.5802
  - AUC-ROC: 0.8422
- **Performance Metrics (After Tuning)**:
  - Accuracy: 0.8112
  - Precision: 0.6931
  - Recall: 0.5147
  - F1-Score: 0.5908
  - AUC-ROC: 0.8629
- **Next Steps**:
  - Address class imbalance (e.g., using SMOTE).
  - Experiment with other advanced models (e.g., LightGBM, CatBoost).

  ## Addressing Class Imbalance (SMOTE)  
- **Technique**: SMOTE (Synthetic Minority Oversampling Technique)  
- **Class Distribution Before SMOTE**:  
  - Non-Churn: 4138  
  - Churn: 1496  
- **Class Distribution After SMOTE**:  
  - Non-Churn: 4138  
  - Churn: 4138
- **Performance Metrics (XGBoost with SMOTE)**:  
  - Accuracy: 77.15%  
  - Precision: 55.81%  
  - Recall: 65.68%  
  - F1-Score: 60.34%  
  - AUC-ROC: 82.25%  
- **Next Steps**:  
  - Experiment with other techniques (e.g., ADASYN, class weights).  
  - Combine SMOTE with hyperparameter tuning.  

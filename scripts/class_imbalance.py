# %% [markdown]
# # Addressing Class Imbalance (SMOTE)
# This notebook applies SMOTE to balance the dataset and trains an XGBoost model on the balanced data.

# %% [markdown]
# ## Load Preprocessed Data

# %%
import pandas as pd

# Load the training and testing sets
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Drop the customerID column (if present)
if 'customerID' in X_train.columns:
    X_train = X_train.drop('customerID', axis=1)
    X_test = X_test.drop('customerID', axis=1)

# Display the shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %% [markdown]
# ## Check Class Distribution

# %%
# Check class distribution
print("Class Distribution in Training Set:")
print(y_train.value_counts())

print("\nClass Distribution in Testing Set:")
print(y_test.value_counts())

# %% [markdown]
# ## Apply SMOTE

# %%
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train.values.ravel())

# Check the new class distribution
print("Class Distribution After SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# %% [markdown]
# ## Train the XGBoost Model on Balanced Data

# %%
from xgboost import XGBClassifier

# Initialize the XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# Train the model on the balanced data
xgb_model.fit(X_train_balanced, y_train_balanced)

print("XGBoost model training complete!")

# %% [markdown]
# ## Evaluate Model Performance

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for AUC-ROC

# Calculate metrics
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_roc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

# Display results
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost Precision: {precision_xgb:.4f}")
print(f"XGBoost Recall: {recall_xgb:.4f}")
print(f"XGBoost F1-Score: {f1_xgb:.4f}")
print(f"XGBoost AUC-ROC: {auc_roc_xgb:.4f}")

# %% [markdown]
# ## Save the Model

# %%
import joblib

# Save the model to a file
joblib.dump(xgb_model, '../models/xgboost_smote.pkl')

print("XGBoost model (with SMOTE) saved to '../models/xgboost_smote.pkl'.")
# %% [markdown]
# # Combining SMOTE with Hyperparameter Tuning
# This notebook combines SMOTE with GridSearchCV to optimize the XGBoost model.

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
# ## Combine SMOTE with GridSearchCV

# %%
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define the pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
])

# Define the parameter grid
param_grid = {
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Perform grid search
grid_search.fit(X_train, y_train.values.ravel())

# Display the best parameters
print("Best Parameters:", grid_search.best_params_)

# %% [markdown]
# ## Train the Tuned XGBoost Model

# %%
# Train the tuned XGBoost model
tuned_xgb_model = grid_search.best_estimator_

print("Tuned XGBoost model training complete!")

# %% [markdown]
# ## Evaluate Model Performance

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Make predictions on the test set
y_pred_tuned_xgb = tuned_xgb_model.predict(X_test)
y_pred_proba_tuned_xgb = tuned_xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for AUC-ROC

# Calculate metrics
accuracy_tuned_xgb = accuracy_score(y_test, y_pred_tuned_xgb)
precision_tuned_xgb = precision_score(y_test, y_pred_tuned_xgb)
recall_tuned_xgb = recall_score(y_test, y_pred_tuned_xgb)
f1_tuned_xgb = f1_score(y_test, y_pred_tuned_xgb)
auc_roc_tuned_xgb = roc_auc_score(y_test, y_pred_proba_tuned_xgb)

# Display results
print(f"Tuned XGBoost Accuracy: {accuracy_tuned_xgb:.4f}")
print(f"Tuned XGBoost Precision: {precision_tuned_xgb:.4f}")
print(f"Tuned XGBoost Recall: {recall_tuned_xgb:.4f}")
print(f"Tuned XGBoost F1-Score: {f1_tuned_xgb:.4f}")
print(f"Tuned XGBoost AUC-ROC: {auc_roc_tuned_xgb:.4f}")

# %% [markdown]
# ## Save the Tuned Model

# %%
import joblib

# Save the tuned model to a file
joblib.dump(tuned_xgb_model, '../models/xgboost_smote_tuned.pkl')

print("Tuned XGBoost model (with SMOTE) saved to '../models/xgboost_smote_tuned.pkl'.")
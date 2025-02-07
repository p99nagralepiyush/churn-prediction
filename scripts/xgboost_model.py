# %% [markdown]
# # Advanced Modeling (XGBoost)
# This notebook trains and evaluates an XGBoost model to predict customer churn.

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
# ## Train an XGBoost Model

# %%
from xgboost import XGBClassifier

# Initialize the XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train, y_train.values.ravel())

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
# ## Hyperparameter Tuning

# %%
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
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
tuned_xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    **grid_search.best_params_
)

tuned_xgb_model.fit(X_train, y_train.values.ravel())

print("Tuned XGBoost model training complete!")

# %% [markdown]
# ## Evaluate the Tuned Model

# %%
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
joblib.dump(tuned_xgb_model, '../models/xgboost_tuned.pkl')

print("Tuned XGBoost model saved to '../models/xgboost_tuned.pkl'.")
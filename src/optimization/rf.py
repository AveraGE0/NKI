import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def split_train_val(df, id_col, seq_col, val_ratio=0.2):
    train_dfs = []
    val_dfs = []

    for id_val, group in df.groupby(id_col):
        n_val_samples = int(len(group) * val_ratio)
        train_df = group.iloc[:-n_val_samples]
        val_df = group.iloc[-n_val_samples:]
        train_dfs.append(train_df)
        val_dfs.append(val_df)

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    return train_df, val_df


ignore_columns = ["dateTime"]
target_column = "CTCAE"
# Identify categorical columns
categorical_columns = ["pat_id", "site", "diagnosis"]


# Load dataset
train_data = pd.read_csv("./data/train_imputed_agg_stats.csv").drop(columns=ignore_columns)
test_data = pd.read_csv("./data/train_imputed_agg_stats.csv").drop(columns=ignore_columns)
# Convert categorical columns to 'category' dtype
le = LabelEncoder()
for col in categorical_columns:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])
# Drop any unnamed columns in the dataframe
train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
train_full = train_data.copy()
train_data, val_data = split_train_val(train_data, id_col='pat_id', seq_col='week_identifier', val_ratio=0.2)
test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]

# Split into features and target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

X_val = val_data.drop(columns=[target_column])
y_val = val_data[target_column]

X_train_full = train_full.drop(columns=[target_column])
y_train_full = train_full[target_column]

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# Define the objective function
def objective(trial):
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 4, 64, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
    max_features = trial.suggest_float('max_features', 0.1, 1)
    
    # Create the model with the sampled hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # Train the model on the training set
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(X_val)

    # Calculate the mean squared error on the validation set
    mse = mean_squared_error(y_val, y_pred)

    # Optuna aims to maximize the objective function, so return the negative MSE
    return -mse

# Create an Optuna study and optimize it
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)

# Print the best hyperparameters
print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Train the final model with the best hyperparameters
best_params = study.best_params
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train_full, y_train_full)

# Save the final model using joblib with compression
joblib.dump(best_model, './models/best_rf_model.pkl', compress=3)

# Evaluate the final model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Validation set MSE:', mse)
print('Validation set MAE:', mae)
print('Validation set R2:', r2)

# Save the evaluation metrics to a file
metrics = {
    'MSE': mse,
    'MAE': mae,
    'R2': r2
}
with open('./results/rf_metrics.json', 'w', encoding="utf-8") as f:
    json.dump(metrics, f)

# Plot feature importance for Random Forest
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = np.array(list(X_train.columns))

plt.figure()
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.savefig('./plots/rf_feature_importance.png')
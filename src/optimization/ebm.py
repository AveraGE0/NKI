import optuna
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import joblib
import json

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

ignore_columns = []
target_column = "CTCAE"
# Load dataset
train_data = pd.read_csv("./data/train_imputed_agg_stats.csv").drop(columns=ignore_columns)
test_data = pd.read_csv("./data/train_imputed_agg_stats.csv").drop(columns=ignore_columns)

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
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01)
    max_leaves = trial.suggest_int('max_leaves', 2, 256)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 100)
    max_bins = trial.suggest_int('max_bins', 32, 512)

    # Create the model with the sampled hyperparameters
    model = ExplainableBoostingRegressor(
        learning_rate=learning_rate,
        max_leaves=max_leaves,
        min_samples_leaf=min_samples_leaf,
        max_bins=max_bins,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    return -mean_squared_error(y_val, y_pred)

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
best_model = ExplainableBoostingRegressor(**best_params, random_state=42)
best_model.fit(X_train_full, y_train_full)
# Save the final model using joblib with compression
joblib.dump(best_model, './models/best_ebm_model.pkl', compress=3)

y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Exclude zero target values for MAPE calculation
non_zero_mask = y_test != 0
mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Test set MSE:', mse)
print('Test set MAE:', mae)
print('Test set MAPE:', mape)
print('Test set R2:', r2)

# Save the evaluation metrics to a file
metrics = {
    'MSE': mse,
    'MAE': mae,
    'MAPE': mape,
    'R2': r2
}
with open('./results/metrics_ebm.json', 'w', encoding="utf-8") as f:
    json.dump(metrics, f)

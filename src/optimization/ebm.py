import optuna
from interpret.glassbox import ExplainableBoostingRegressor
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

categorical_columns = ["pat_id", "sex", "smoker", "drinker"]
ignore_columns = ["dateTime", "week_indicator"]
target_column = "CTCAE"
# Load dataset
train_data = pd.read_csv("./data/train_imputed_agg_stats.csv").drop(columns=ignore_columns)
test_data = pd.read_csv("./data/test_imputed_agg_stats.csv").drop(columns=ignore_columns)

for col in categorical_columns:
    train_data[col] = train_data[col].astype(str)
    test_data[col] = test_data[col].astype(str)

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
best_model = ExplainableBoostingRegressor(**best_params, random_state=42, feature_names = X_train_full.columns.tolist())
best_model.fit(X_train_full, y_train_full)
# Save the final model using joblib with compression
joblib.dump(best_model, './models/best_ebm_model.pkl', compress=3)

# make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# calculate scores
train_mse = mean_squared_error(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Calculate evaluation metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# Print the evaluation metrics
print('Test set MSE:', test_mse)
print('Test set MAE:', test_mae)
print('Test set R2:', test_r2)

# Print the evaluation metrics
print('Train set MSE:', train_mse)
print('Train set MAE:', train_mae)
print('Train set R2:', train_r2)
# Save the evaluation metrics to a file
metrics = {
    'Train_MSE': train_mse,
    'Train_MAE': train_mae,
    'Train_R2': train_r2,
    'Test_MSE': test_mse,
    'Test_MAE': test_mae,
    'Test_R2': test_r2
}

predictions = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred_test
})
predictions.to_csv('./results/ebm_predictions.csv', index=False)

with open('./results/metrics_ebm.json', 'w', encoding="utf-8") as f:
    json.dump(metrics, f)

# Plot feature contributions for EBM
#from interpret import show
#import matplotlib.pyplot as plt

#ebm_global = best_model.explain_global(name='EBM Feature Importances')

# Render the plot
#fig = show(ebm_global)

# Save the plot to the filesystem
#fig.write_image("./plots/ebm_feature_contributions.png")
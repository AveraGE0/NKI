import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import json


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
best_model.fit(X_train, y_train)

# Save the final model using joblib with compression
joblib.dump(best_model, './models/best_rf_model.pkl', compress=3)

# Evaluate the final model on the validation set
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

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
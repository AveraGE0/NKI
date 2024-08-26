"""Script to train a (unoptimized) ebm model. Can mainly be used for quick testing
and for orientation when making new scripts"""
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.evaluation.adjusted_r2 import adjusted_r2_score
from src.evaluation.model_analysis import plot_errors, plot_predicted_scores, groupwise_errors
from src.modeling.model_manager import create_model_directory, unique_model_name
from src.config_loader import train_path, test_path, categorical_columns, ignore_columns, target_column
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


# Load dataset
train_data = pd.read_csv(train_path).drop(columns=ignore_columns)
test_data = pd.read_csv(test_path).drop(columns=ignore_columns)

for col in categorical_columns:
    train_data[col] = train_data[col].astype(str)
    test_data[col] = test_data[col].astype(str)

# Drop any unnamed columns in the DataFrame
train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
train_full = train_data.copy()
train_data, val_data = split_train_val(
    train_data,
    id_col='pat_id',
    seq_col='week_identifier',
    val_ratio=0.2
)
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

# Create model directory
model_name = unique_model_name("test_ebm")
create_model_directory(model_name, {
    f"{model_name}": "Un-optimized EBM model with standard parameters",
    "Parameters": ", ".join(X_test.columns)
})

# Create the model with the sampled hyperparameters
model = ExplainableBoostingRegressor(
    #learning_rate=learning_rate,
    #max_leaves=max_leaves,
    #min_samples_leaf=min_samples_leaf,
    #max_bins=max_bins,
    random_state=42,
    feature_names = X_train_full.columns.tolist()
)

model.fit(X_train, y_train)

# Save the final model using joblib with compression
joblib.dump(model, f'./models/{model_name}/best_ebm_model.pkl', compress=3)

metrics = {}
# make evaluation
for eval_set_name, X, y in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test)
]:
    y_pred = model.predict(X)
    df_data: pd.DataFrame = X.copy()
    df_data["actual"] = y
    df_data["predicted"] = y_pred
    df_data["onegroup"] = 1

    # variation per patient (within)
    df_within_patient_errors = groupwise_errors(df_data, "actual", "predicted", ["pat_id"])
    with open(f'./models/{model_name}/{eval_set_name}_within_patient.json', 'w', encoding="utf-8") as f:
        json.dump(df_within_patient_errors, f)
    # variation between patients
    df_between_patient_errors = groupwise_errors(df_data, "actual", "predicted", ["onegroup"])
    with open(f'./models/{model_name}/{eval_set_name}_between_patients.json', 'w', encoding="utf-8") as f:
        json.dump(df_between_patient_errors, f)
    # variation within cancer types
    #df_within_cancer_errors = groupwise_errors(df_data, "actual", "predicted", ["diagnosis"])
    #with open(f'./models/{model_name}/{eval_set_name}_cancer_prediction_variation.json', 'w', encoding="utf-8") as f:
    #    json.dump(df_within_cancer_errors, f)
    
    # calculate scores
    metrics.update({
        f"{eval_set_name}_mse": mean_squared_error(y, y_pred),
        f"{eval_set_name}_mae": mean_absolute_error(y, y_pred),
        f"{eval_set_name}_r2": r2_score(y, y_pred),
        f"{eval_set_name}_adj_r2": adjusted_r2_score(X.shape[0], X.shape[1], r2_score(y, y_pred))
    })

    with open(f'./models/{model_name}/{eval_set_name}_metrics.json', 'w', encoding="utf-8") as f:
        json.dump(metrics, f)

# Plot feature contributions for EBM
from interpret import show
import matplotlib.pyplot as plt

ebm_global = model.explain_global(name='EBM Feature Importances')

# Extracting feature names and their corresponding importances
feature_names = ebm_global.data()['names']
importances = ebm_global.data()['scores']

# Create a bar plot using Matplotlib
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('EBM Feature Importances')

# Save the plot to a specific path
plt.savefig(f'./models/{model_name}/ebm_feature_importances.png')
plt.close()
"""Script to train a (unoptimized) ebm model. Can mainly be used for quick testing
and for orientation when making new scripts"""
import pandas as pd
import joblib
import json
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.evaluation.adjusted_r2 import adjusted_r2_score
from src.evaluation.model_analysis import plot_errors, plot_predicted_scores, groupwise_errors
from src.evaluation.plots import pat_eval_plot
from src.modeling.model_manager import create_model_directory, unique_model_name, add_description
from src.config_loader import train_path_regression as train_path
from src.config_loader import test_path_regression as test_path
from src.config_loader import categorical_columns_regression as categorical_columns
from src.config_loader import ignore_columns_regression as ignore_columns
from src.config_loader import target_column_regression as target_column
from src.config_loader import feature_types
from src.config_loader import id_column, seq_column
from src.evaluation.plots import plot_feature_importance
import matplotlib.pyplot as plt
from src.dataset import load_data, split_data


df_train, df_val, df_test, train_ignore, val_ignore, test_ignore = load_data(
    train_path=train_path,
    test_path=test_path,
    id_col=id_column,
    seq_col=seq_column,
    ignore_cols=ignore_columns,
    categorical_cols=categorical_columns
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
    df_train,
    df_val,
    df_test,
    target_col=target_column
)

# Create model directory
model_name = unique_model_name("test_ebm")
create_model_directory(model_name, {
    f"{model_name}": "Un-optimized EBM model with standard parameters",
    "Used Features": ", ".join(X_test.columns)
})

# Create the model with the sampled hyperparameters
model = ExplainableBoostingRegressor(
    #learning_rate=learning_rate,
    #max_leaves=max_leaves,
    #min_samples_leaf=min_samples_leaf,
    #max_bins=max_bins,
    random_state=42,
    feature_names = X_train.columns.tolist(),
    feature_types=[feature_types[feature] if feature in feature_types.keys() else None for feature in X_train.columns.tolist()]
)

model.fit(X_train, y_train)
# save the parameters in the description
add_description(
    f'./models/{model_name}/model_description.md',
    headline="Model Parameters",
    content='\n'.join(f'{key}: {value}' for key, value in model.get_params().items())
)

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

    selected = ["pat_id", "actual", "predicted"]
    if "diagnosis" in df_data.columns:
        selected += ["diagnosis"]

    df_data[selected].to_csv(f'./models/{model_name}/{eval_set_name}_prediction.csv')
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

# feature importance
fig = plot_feature_importance(model)
fig.savefig(f'./models/{model_name}/ebm_feature_importances.png', dpi=300)
plt.close()


with open(f'./models/{model_name}/train_within_patient.json', 'r', encoding="utf-8") as f:
    train = json.load(f)
with open(f'./models/{model_name}/val_within_patient.json', 'r', encoding="utf-8") as f:
    val = json.load(f)
with open(f'./models/{model_name}/test_within_patient.json', 'r', encoding="utf-8") as f:
    test = json.load(f)

df_train = pd.DataFrame(train).T
df_val = pd.DataFrame(val).T
df_test = pd.DataFrame(test).T

## Combine into a single DataFrame
df_combined = pd.concat([df_train.add_suffix('_train'),
                         df_val.add_suffix('_val'),
                         df_test.add_suffix('_test')], axis=1)
pat_error_fig = pat_eval_plot(df_combined, x_name="mse", y_name="error_std")
pat_error_fig.savefig(f'./models/{model_name}/within_patient_error.png')
plt.close()

# plot errors
df_predictions_test = pd.read_csv(f'./models/{model_name}/test_prediction.csv')
df_predictions_test["test"] = 1
df_predictions_train = pd.read_csv(f'./models/{model_name}/train_prediction.csv')
df_predictions_train["train"] = 1

plot_errors(df_predictions_test, "actual", "predicted", "test", model_path=f'./models/{model_name}')
plot_errors(df_predictions_train, "actual", "predicted", "train", model_path=f'./models/{model_name}')

plot_predicted_scores(df_predictions_test, "actual", "predicted", "test", model_path=f'./models/{model_name}')
plot_predicted_scores(df_predictions_train, "actual", "predicted", "train", model_path=f'./models/{model_name}')

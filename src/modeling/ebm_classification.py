from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from src.evaluation.model_analysis import calculate_metrics_per_id, plot_classification_metrics, groupwise_metrics
from src.modeling.model_manager import create_model_directory, unique_model_name, add_description
from src.config_loader import train_path_classification as train_path
from src.config_loader import test_path_classification as test_path
from src.config_loader import categorical_columns_classification as categorical_columns
from src.config_loader import ignore_columns_classification as ignore_columns
from src.config_loader import target_column_classification as target_column
from src.config_loader import feature_types
from src.evaluation.plots import pat_eval_classification_plot
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from src.dataset import split_train_val

# Load dataset
train_data = pd.read_csv(train_path).drop(columns=ignore_columns)
test_data = pd.read_csv(test_path).drop(columns=ignore_columns)

for col in categorical_columns:
    train_data[col] = train_data[col].astype(str)
    test_data[col] = test_data[col].astype(str)

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
model_name = unique_model_name("test_ebm_classifier")
create_model_directory(model_name, {
    f"{model_name}": "Un-optimized EBM classifier with standard parameters",
    "Parameters": ", ".join(X_test.columns)
})

# Create the Explainable Boosting Classifier (EBC)
model = ExplainableBoostingClassifier(
    random_state=42,
    feature_names=X_train_full.columns.tolist(),
    feature_types=[feature_types[feature] if feature in feature_types.keys() else None for feature in X_train_full.columns.tolist()]
)

model.fit(X_train, y_train)

# Save the model
add_description(
    f'./models/{model_name}/model_description.md',
    headline="Parameters",
    content='\n'.join(f'{key}: {value}' for key, value in model.get_params().items())
)
joblib.dump(model, f'./models/{model_name}/best_ebm_classifier.pkl', compress=3)

metrics = {}
# Make evaluation
for eval_set_name, X, y in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test)
]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Probability for positive class

    # Save predictions
    df_data = X.copy()
    df_data["actual"] = y
    df_data["predicted"] = y_pred
    df_data["proba"] = y_proba
    df_data["onegroup"] = 1

    selected = ["pat_id", "actual", "predicted", "proba"]
    if "diagnosis" in df_data.columns:
        selected += ["diagnosis"]

    df_data[selected].to_csv(f'./models/{model_name}/{eval_set_name}_prediction.csv')

    # Calculate metrics
    metrics.update({
        f"{eval_set_name}_accuracy": accuracy_score(y, y_pred),
        f"{eval_set_name}_precision": precision_score(y, y_pred, average='weighted'),
        f"{eval_set_name}_recall": recall_score(y, y_pred, average='weighted'),
        f"{eval_set_name}_f1": f1_score(y, y_pred, average='weighted')
    })

    with open(f'./models/{model_name}/{eval_set_name}_metrics.json', 'w', encoding="utf-8") as f:
        json.dump(metrics, f)
    
    # variation per patient (within)
    df_within_patient_errors = groupwise_metrics(df_data, "actual", "predicted", "pat_id")
    with open(f'./models/{model_name}/{eval_set_name}_within_patient.json', 'w', encoding="utf-8") as f:
        json.dump(df_within_patient_errors, f)
    # variation between patients
    df_between_patient_errors = groupwise_metrics(df_data, "actual", "predicted", "onegroup")
    with open(f'./models/{model_name}/{eval_set_name}_between_patients.json', 'w', encoding="utf-8") as f:
        json.dump(df_between_patient_errors, f)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({eval_set_name})')
    plt.savefig(f'./models/{model_name}/{eval_set_name}_confusion_matrix.png')
    plt.close()

# Plot feature importances
ebm_global = model.explain_global(name='EBM Feature Importances')
feature_names = ebm_global.data()['names']
importances = ebm_global.data()['scores']

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('EBM Feature Importances')
plt.savefig(f'./models/{model_name}/ebm_feature_importances.png')
plt.close()


#df_metrics = calculate_metrics_per_id(df_data, id_col='id', actual_col='actual', prediction_col='predicted')
#pat_metric_fig = plot_classification_metrics(df_metrics)
#pat_metric_fig.savefig(f'./models/{model_name}/patient_metrics.png')
#pat_metric_fig.close()

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
fig = pat_eval_classification_plot(df_combined, 'precision')
fig.savefig(f'./models/{model_name}/within_patient_metrics.png')
plt.close()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_feature_importance(model) -> Figure:
    # Plot feature contributions for EBM
    ebm_global = model.explain_global(name='EBM Feature Importances')

    # Extracting feature names and their corresponding importances
    feature_names = ebm_global.data()['names']
    importances = ebm_global.data()['scores']

    # Create a bar plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.barh(feature_names, importances)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('EBM Feature Importances')

    plt.tight_layout()
    return fig



def pat_eval_plot(df_combined: pd.DataFrame, x_name: str, y_name: str) -> plt.Figure:
# Assuming train, val, and test dictionaries are defined as described above

# Convert dictionaries to DataFrames
#df_train = pd.DataFrame(train).T
#df_val = pd.DataFrame(val).T
#df_test = pd.DataFrame(test).T

## Combine into a single DataFrame
#df_combined = pd.concat([df_train.add_suffix('_train'),
#                         df_val.add_suffix('_val'),
#                         df_test.add_suffix('_test')], axis=1)

    fig, ax = plt.subplots(figsize=(15, 6))

    ids = df_combined.index
    x = np.arange(len(ids))  # the label locations

    # Plot train means with std error bars
    ax.errorbar(x, df_combined[f'{x_name}_train'], yerr=df_combined[f'{y_name}_train'], fmt='o', label=f'Train {x_name}', capsize=3)

    # Plot val means with std error bars
    ax.errorbar(x, df_combined[f'{x_name}_val'], yerr=df_combined[f'{y_name}_val'], fmt='s', label=f'Validation {x_name}', capsize=3)

    # Plot test means with std error bars
    ax.errorbar(x, df_combined[f'{x_name}_test'], yerr=df_combined[f'{y_name}_test'], fmt='^', label=f'Test {x_name}', capsize=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('ID')
    ax.set_ylabel(x_name)
    ax.set_title('Mean and Standard Deviation for Train, Validation, and Test Groups')
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90)
    ax.legend()

    fig.tight_layout()
    return fig
    #fig.savefig()
    #plt.show()


def pat_eval_classification_plot(df_combined: pd.DataFrame, metric_name: str) -> plt.Figure:
    """
    Plot precision, recall, accuracy, or F1 for each ID across train, validation, and test sets.

    Parameters:
    df_combined (pd.DataFrame): DataFrame containing metrics for each ID across train, validation, and test sets.
    metric_name (str): The metric to plot (e.g., 'precision', 'recall', 'accuracy', 'f1').

    Returns:
    plt.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    ids = df_combined.index
    x = np.arange(len(ids))  # the label locations

    # Plot train metrics
    ax.plot(x, df_combined[f'{metric_name}_train'], 'o', label=f'Train {metric_name.capitalize()}', markersize=5)

    # Plot validation metrics
    ax.plot(x, df_combined[f'{metric_name}_val'], 's', label=f'Validation {metric_name.capitalize()}', markersize=5)

    # Plot test metrics
    ax.plot(x, df_combined[f'{metric_name}_test'], '^', label=f'Test {metric_name.capitalize()}', markersize=5)

    # Add labels, title, and legend
    ax.set_xlabel('ID')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'{metric_name.capitalize()} for Train, Validation, and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90)
    ax.legend()

    fig.tight_layout()
    return fig
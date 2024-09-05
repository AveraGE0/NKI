"""
This module provides functionality to evaluate a models performance.
Specifically it provides functions to evaluate the prediction (errors) on
different (sub) population in order to see where the model under-performs
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def groupwise_errors(df_data: pd.DataFrame, actual_col: str, prediction_col: str, grouping_cols: list) -> dict:
    """Calculates the errors of a specific group of the data. Errors include absolute error and squared error
    with their mean and standard deviation.

    Args:
        df_data (pd.DataFrame): DataFrame containing at least prediction, ground truth and grouping colunms
        actual_col (str): name of the column containing the true values
        prediction_col (str): name of the column containing the predicted values
        grouping_cols (list): list of the names of the column containing the grouping columns

    Returns:
        dict: a row is one group, where each value is one metric
    """
    df_data["error"] = (df_data[actual_col] - df_data[prediction_col]).abs()
    df_data["sq_error"] = df_data["error"] * df_data["error"]
    df_data = df_data.groupby(by=grouping_cols)

    group_stats_df = pd.DataFrame({
        "mae": df_data["error"].mean(),
        "error_std": df_data["error"].std(),
        "mse": df_data["sq_error"].mean(),
        "sq_error_std": df_data["sq_error"].std()
    })
    # return scores in dict
    scores_dict = group_stats_df.to_dict(orient="index")
    return scores_dict


def plot_predicted_scores(df_data: pd.DataFrame, actual_col: str, prediction_col: str, grouping_col: str, model_path: str):
    for group in df_data[grouping_col].unique():
        df_group = df_data[df_data[grouping_col] == group]

        # Create a figure with a GridSpec to manage the subplots
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, figure=fig)

        # Main scatter plot in the center
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_main.scatter(df_group[actual_col], df_group[prediction_col], alpha=0.7)
        ax_main.set_xlabel("Actual value")
        ax_main.set_ylabel("Predicted value")
        ax_main.set_title(f'Scatterplot of actual vs predicted value ({grouping_col}={group})')

        # Distribution plot for the x-axis (Actual value)
        ax_xdist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        sns.kdeplot(x=df_group[actual_col], ax=ax_xdist, color='blue', fill=True)
        ax_xdist.get_xaxis().set_visible(False)
        ax_xdist.set_ylabel('Density')

        # Distribution plot for the y-axis (Predicted value)
        ax_ydist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        sns.kdeplot(y=df_group[prediction_col], ax=ax_ydist, color='orange', fill=True)
        ax_ydist.get_yaxis().set_visible(False)
        ax_ydist.set_xlabel('Density')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{model_path}/{grouping_col}={group}_pred_vs_actual_with_density.png')
        plt.close()
    

def plot_errors(df_data: pd.DataFrame, actual_col: str, prediction_col: str, grouping_col: str, model_path: str):
    for group in df_data[grouping_col].unique():
        df_group = df_data[df_data[grouping_col]==group]
        plt.scatter(
            df_group[actual_col],
            df_group[actual_col] - df_group[prediction_col]
        )
        plt.xlabel("True value")
        plt.ylabel("Error")
        plt.title('Scatterplot of actual vs error value')

        # Save the plot to a specific path
        plt.savefig(f'{model_path}/{grouping_col}={group}_score_vs_error.png')
        plt.close()


# Assuming you have a DataFrame `df_data` with columns: 'id', 'actual', and 'predicted'
def calculate_metrics_per_id(df_data: pd.DataFrame, id_col: str, actual_col: str, prediction_col: str):
    metrics = []

    for _id in df_data[id_col].unique():
        df_group = df_data[df_data[id_col] == _id]
        
        precision = precision_score(df_group[actual_col], df_group[prediction_col], zero_division=0)
        recall = recall_score(df_group[actual_col], df_group[prediction_col], zero_division=0)
        accuracy = accuracy_score(df_group[actual_col], df_group[prediction_col])
        f1 = f1_score(df_group[actual_col], df_group[prediction_col], zero_division=0)
        
        metrics.append({
            'id': _id,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        })

    return pd.DataFrame(metrics)


def plot_classification_metrics(df_metrics: pd.DataFrame):
    """
    Plot precision, recall, accuracy, and F1 score for each ID and return the figure.

    Parameters:
    df_metrics (pd.DataFrame): DataFrame containing 'id', 'precision', 'recall', 'accuracy', and 'f1' columns.

    Returns:
    plt.Figure: The matplotlib figure object.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the metrics
    ax.plot(df_metrics['id'], df_metrics['precision'], label='Precision', marker='o')
    ax.plot(df_metrics['id'], df_metrics['recall'], label='Recall', marker='o')
    ax.plot(df_metrics['id'], df_metrics['accuracy'], label='Accuracy', marker='o')
    ax.plot(df_metrics['id'], df_metrics['f1'], label='F1 Score', marker='o')

    # Add labels and title
    ax.set_xlabel('ID')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, Accuracy, and F1 Score for each ID')
    ax.legend()

    # Rotate x-axis labels if there are many IDs
    ax.set_xticks(df_metrics['id'])
    ax.set_xticklabels(df_metrics['id'], rotation=90)

    # Adjust layout to avoid overlap
    fig.tight_layout()

    # Return the figure
    return fig


def groupwise_metrics(df_data, actual_col, predicted_col, group_col: str):
    """
    Calculate precision, recall, accuracy, and F1 score for each group defined by group_cols.

    Parameters:
    df_data (pd.DataFrame): The input DataFrame containing actual and predicted values.
    actual_col (str): Column name of the actual values.
    predicted_col (str): Column name of the predicted values.
    group_cols (list): List of columns to group by (e.g., ['pat_id']).

    Returns:
    dict: A dictionary where keys are the group ids and values are dictionaries of classification metrics.
    """
    metrics_per_group = {}

    # Group the data by the specified columns
    grouped = df_data.groupby(group_col)

    # Iterate over each group and calculate metrics
    for group_id, group_df in grouped:
        # Calculate metrics for the current group
        precision = precision_score(group_df[actual_col], group_df[predicted_col], zero_division=0, average='macro')
        recall = recall_score(group_df[actual_col], group_df[predicted_col], zero_division=0, average='macro')
        accuracy = accuracy_score(group_df[actual_col], group_df[predicted_col])
        f1 = f1_score(group_df[actual_col], group_df[predicted_col], zero_division=0, average='macro')

        # Store the metrics in a dictionary for this group
        metrics_per_group[group_id] = {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        }

    return metrics_per_group
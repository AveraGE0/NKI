"""
This module provides functionality to evaluate a models performance.
Specifically it provides functions to evaluate the prediction (errors) on
different (sub) population in order to see where the model under-performs
"""
import pandas as pd
import matplotlib.pyplot as plt


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
        df_group = df_data[df_data[grouping_col]==group]
        plt.scatter(
            df_group[actual_col],
            df_group[prediction_col]
        )
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.title('Scatterplot of actual vs predicted value')

        # Save the plot to a specific path
        plt.savefig(f'{model_path}/{group}_pred_vs_actual.png')
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
        plt.savefig(f'{model_path}/{group}_score_vs_error.png')
        plt.close()

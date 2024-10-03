"""Module for calculating metric scores on model and datasets"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.evaluation.adjusted_r2 import adjusted_r2_score
from src.evaluation.model_analysis import groupwise_errors


def make_save_metrics(model_path: str, eval_sets: list[tuple], y_col: str, y_pred_col: str) -> None:
    metrics = {}
    # make evaluation
    for eval_set_name, df_data in eval_sets:
        df_data["onegroup"] = 1

        # variation per patient (within)
        df_within_patient_errors = groupwise_errors(df_data, "actual", "predicted", ["pat_id"])
        with open(f'{model_path}/{eval_set_name}_within_patient.json', 'w', encoding="utf-8") as f:
            json.dump(df_within_patient_errors, f)
        # variation between patients
        df_between_patient_errors = groupwise_errors(df_data, "actual", "predicted", ["onegroup"])
        with open(f'{model_path}/{eval_set_name}_between_patients.json', 'w', encoding="utf-8") as f:
            json.dump(df_between_patient_errors, f)
        # variation within cancer types
        #df_within_cancer_errors = groupwise_errors(df_data, "actual", "predicted", ["diagnosis"])
        #with open(f'{model_path}/{eval_set_name}_cancer_prediction_variation.json', 'w', encoding="utf-8") as f:
        #    json.dump(df_within_cancer_errors, f)

        # calculate scores
        metrics.update({
            f"{eval_set_name}_mse": mean_squared_error(df_data[y_col], df_data[y_pred_col]),
            f"{eval_set_name}_mae": mean_absolute_error(df_data[y_col], df_data[y_pred_col]),
            f"{eval_set_name}_r2": r2_score(df_data[y_col], df_data[y_pred_col]),
            f"{eval_set_name}_adj_r2": adjusted_r2_score(len(df_data), len(df_data.columns), r2_score(df_data[y_col], df_data[y_pred_col]))
        })

        with open(f'{model_path}/{eval_set_name}_metrics.json', 'w', encoding="utf-8") as f:
            json.dump(metrics, f)

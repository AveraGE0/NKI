"""Module to test the model analysis module"""
import os
import pandas as pd
from src.evaluation import model_analysis


def test_groupwise_errors():
    df_data = pd.DataFrame().from_dict({
        "actual": [0.0, 2.0, 0.0, 1.0],
        "prediction": [0.0, 1.0, 0.0, 0.0],
        "group_indicator": [15001, 15002, 15001, 15002]
    })
    scores = model_analysis.groupwise_errors(
        df_data,
        actual_col="actual",
        prediction_col="prediction",
        grouping_cols=["group_indicator"]
    )
    assert scores[15001]["mae"] == 0
    assert scores[15001]["error_std"] == 0
    assert scores[15001]["mse"] == 0
    assert scores[15001]["sq_error_std"] == 0

    assert scores[15002]["mae"] == 1.0
    assert scores[15002]["error_std"] == 0.0
    assert scores[15002]["mse"] == 1.0
    assert scores[15002]["sq_error_std"] == 0.0


def test_plot_pred_vs_actual():
    df_data = pd.DataFrame().from_dict({
        "actual": [0.0, 2.0, 0.0, 1.0],
        "prediction": [0.0, 1.0, 0.0, 0.0],
        "group_indicator": [15001, 15002, 15001, 15002]
    })
    model_analysis.plot_predicted_scores(
        df_data,
        "actual",
        "prediction",
        "group_indicator",
        "./tests/tmp_test"
    )
    assert os.path.isfile("./tests/tmp_test/15001_pred_vs_actual.png")
    assert os.path.isfile("./tests/tmp_test/15002_pred_vs_actual.png")

    # cleanup
    os.remove("./tests/tmp_test/15001_pred_vs_actual.png")
    os.remove("./tests/tmp_test/15002_pred_vs_actual.png")


def test_plot_error_vs_actual():
    df_data = pd.DataFrame().from_dict({
        "actual": [0.0, 2.0, 0.0, 1.0],
        "prediction": [0.0, 1.0, 0.0, 0.0],
        "group_indicator": [15001, 15002, 15001, 15002]
    })
    model_analysis.plot_errors(
        df_data,
        "actual",
        "prediction",
        "group_indicator",
        "./tests/tmp_test"
    )
    assert os.path.isfile("./tests/tmp_test/15001_score_vs_error.png")
    assert os.path.isfile("./tests/tmp_test/15002_score_vs_error.png")

    # cleanup
    os.remove("./tests/tmp_test/15001_score_vs_error.png")
    os.remove("./tests/tmp_test/15002_score_vs_error.png")
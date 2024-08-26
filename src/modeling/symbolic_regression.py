import logging
import os
from datetime import datetime
from pysr import PySRRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.time_series_splitting import train_test_date_split


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

def run_pysr(
        X_train: np.array,
        y_train: np.array,
        unary_operators: list["str"],
        binary_operators: list["str"],
        model_config: dict
    ) -> PySRRegressor:
    model = PySRRegressor(
        niterations=20,
        populations=20,
        population_size=100,
        ncycles_per_iteration=100,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constraints={'^': (-1, 1)},
        nested_constraints={
            "^": {"^": 1, "cube": 1, "exp": 0},
            "cube": {"^": 1, "cube": 1, "exp": 0},
            "exp": {"^": 1, "cube": 1, "exp": 0},
        },
        maxsize=20,
        elementwise_loss="loss(x, y) = (x - y)^2",
        tempdir="./pysr_temp",
        temp_equation_file=True,
        delete_tempfiles=False,
    )
    # Fit the model
    model.fit(X_train, y_train)
    # Print the best model
    print(model)
    return model


def calculate_errors(
        model_function: PySRRegressor,
        X_test:np.ndarray,
        y_test: np.ndarray
    ) -> tuple[float, float]:
    """
    Calculate MSE and MAE given a model function and test data.

    Args:
        model_function (function): The function representing the model found by the GP.
        X_test (np.ndarray): The test set features.
        y_test (np.ndarray): The test set targets.

    Returns:
        tuple: MSE and MAE of the model on the test set.
    """
    # Predict using the model function
    y_pred = model_function(X_test)

    # Calculate MSE and MAE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae


def get_best_models_df(
        model: PySRRegressor,
        X_train,
        y_train,
        X_test,
        y_test
    ) -> None:
    # Retrieve the best performing functions
    best_equations = model.equations_

    # Sort by loss to get the best performing functions
    best_equations_sorted = best_equations.sort_values(by="loss")

    # Print the top 5 best performing functions with their errors
    logger.info("Top 10 Best Performing Functions:")
    logger.info(
        "%-40s %-15s %-15s %-15s %-15s",
        "Formula", "Train MSE", "Train MAE", "Test MSE", "Test MAE"
    )
    logger.info("-" * 120)

    results = []

    for i in range(min(10, len(best_equations_sorted))):
        equation = best_equations_sorted.iloc[i]
        formula = equation['equation']
        model_function = equation['lambda_format']

        # Calculate errors for train, validation, and test sets
        train_mse, train_mae = calculate_errors(model_function, X_train, y_train)
        test_mse, test_mae = calculate_errors(model_function, X_test, y_test)

        logger.info(
            "%-40s %-15.5f %-15.5f %-15.5f %-15.5f", 
            formula, train_mse, train_mae, test_mse, test_mae
        )


        results.append({
            "Formula": formula,
            "Train MSE": train_mse,
            "Train MAE": train_mae,
            "Test MSE": test_mse,
            "Test MAE": test_mae
        })

        # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == '__main__':
    used_features = ["respiration_rate", "bpm", "temp_sk1"]
    df_train, df_test = pd.read_csv('./data/train_data.csv'), pd.read_csv('./data/test_data.csv')
    df_train, df_val = train_test_date_split(df_train, 0.9, 'date', ['id'])

    X_train = df_train[used_features].values
    y_train = (df_train["bpm"].values + df_train["temp_sk1"]) / 2
    X_test = df_test[used_features].values
    y_test = (df_test["bpm"].values + df_test["temp_sk1"]) / 2
    # Define the function set
    binary_operators = ["+", "-", "*", "/", "^"]
    unary_operators = ["cos", "exp", "sin", "cube", "sqrt"]
    forms_simple = run_pysr(X_train, y_train, unary_operators, binary_operators, {})
    df_results = get_best_models_df(
        forms_simple,
        X_train, y_train,
        X_test, y_test
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"best_models_results_{timestamp}.csv"
    df_results.to_csv(os.path.join('./pysr_results', results_filename), index=False)
    

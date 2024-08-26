import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestRegressor


def main():
    # Define the CLI
    parser = argparse.ArgumentParser(description="Train and evaluate a regression model.")
    parser.add_argument("train_data_path", type=str, help="Path to the CSV file containing the training data.")
    parser.add_argument("test_data_path", type=str, help="Path to the CSV file containing the test data.")
    parser.add_argument("target_column", type=str, help="The target column in the data files.")
    parser.add_argument("model", type=str, choices=["RF", "EBM"], help="The model that is being trained")
    parser.add_argument("results_path", type=str, help="Path to save the results data frame and predictions.")
    parser.add_argument("--ignore_columns", nargs="+", help="Columns to ignore during imputation.")
    args = parser.parse_args()

    # Load the data
    train_data = pd.read_csv(args.train_data_path).drop(columns=args.ignore_columns)
    test_data = pd.read_csv(args.test_data_path).drop(columns=args.ignore_columns)
    # Drop any unnamed columns in the dataframe
    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
    # Split into features and target
    X_train = train_data.drop(columns=[args.target_column])
    y_train = train_data[args.target_column]
    X_test = test_data.drop(columns=[args.target_column])
    y_test = test_data[args.target_column]

    models = {
        "RF": RandomForestRegressor(),
        "EBM": ExplainableBoostingClassifier()
    }
    regressor = models[args.model]
    # Train the regression model
    regressor.fit(X_train, y_train)

    # Make predictions
    train_predictions = pd.to_numeric(pd.Series(regressor.predict(X_train)), errors='coerce').to_numpy()
    test_predictions = pd.to_numeric(pd.Series(regressor.predict(X_test)), errors='coerce').to_numpy()
    # Evaluate the model
    mse_train = mean_squared_error(y_train, train_predictions)
    mae_train = mean_absolute_error(y_train, train_predictions)
    r2_train = r2_score(y_train, train_predictions)

    mse_test = mean_squared_error(y_test, test_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)
    r2_test = r2_score(y_test, test_predictions)

    # Save the performance scores to a DataFrame
    results_df = pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "MSE": [mse_train, mse_test],
        "MAE": [mae_train, mae_test],
        "R2": [r2_train, r2_test]
    })

    # Save the results DataFrame
    results_df.to_csv(f"{args.results_path}/{args.model}_performance_scores.csv", index=False)

    # Save the predictions and actual scores
    train_results = pd.DataFrame({
        "Actual": y_train,
        "Predicted": train_predictions
    })
    test_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": test_predictions
    })

    train_results.to_csv(f"{args.results_path}/{args.model}_train_predictions.csv", index=False)
    test_results.to_csv(f"{args.results_path}/{args.model}_test_predictions.csv", index=False)
    # Save the trained model
    model_path = f"{args.results_path}/{args.model}_trained_model.pkl"
    joblib.dump(regressor, model_path)

    print("Training and evaluation complete. Results saved.")


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.imputation import ImputationEvaluation, LinearImputation, RandomForestIterativeImputation, RandomForestImputation, KNNImputation


# Custom JSON encoder for NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    parser = argparse.ArgumentParser(description="Evaluate different imputation methods.")
    parser.add_argument("data_path", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("--grouping_columns", nargs="+", help="Columns to use for grouping data.")
    parser.add_argument("--ignore_columns", nargs="+", help="Columns to ignore during imputation.")
    parser.add_argument("--test_drop_amount", type=float, default=0.05, help="Amount of data to drop for testing.")
    parser.add_argument("--tested_sequence_lengths", nargs="+", type=int, default=[1], help="Sequence lengths to test.")
    parser.add_argument("--methods", nargs="+", choices=["LINEAR", "KNN", "RF", "MICE"], default=["linear"], help="Imputation methods to evaluate.")
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path, index_col=0)
    # Verify it does not contain NA values (only for testing)
    assert not data.isnull().values.any(), "The data you are trying to test contains NaN values! Aborting imputation test."

    # Initialize imputation methods
    methods = []
    for method in args.methods:
        if method == "LINEAR":
            methods.append(LinearImputation())
        elif method == "RF":
            methods.append(RandomForestImputation())
        elif method == "KNN":
            methods.append(KNNImputation())
        elif method == "MICE":
            methods.append(RandomForestIterativeImputation())
        # Add other imputation methods here as needed

    # Create ImputationEvaluation instance
    evaluator = ImputationEvaluation(
        test_data=data,
        grouping_columns=args.grouping_columns,
        ignore_columns=args.ignore_columns,
        test_drop_amount=args.test_drop_amount,
        tested_sequence_lengths=args.tested_sequence_lengths
    )

    # Test models
    evaluator.test_models(methods)

    # Get results
    results = evaluator.get_results()
    results_group = evaluator.get_groupwise_results()
    

    results.to_csv(Path(args.data_path).parent / "results.csv")
    # Save the dictionary to a csv
    results_group.to_csv(Path(args.data_path).parent / "group_results.csv")
    # Print results
    print(results)

if __name__ == "__main__":
    main()

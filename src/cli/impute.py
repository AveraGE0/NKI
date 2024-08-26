"""Module to run imputation from CLI"""

import argparse
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from src.imputation import Trainable, LinearImputation, RandomForestIterativeImputation, RandomForestImputation, KNNImputation


def get_model_name(method, group_name) -> str:
    return f"model_impute_{method}_{str(group_name).replace('(', '').replace(')', '').replace(',', '_')}"

def main():
    parser = argparse.ArgumentParser(description="Evaluate different imputation methods.")
    parser.add_argument("data_path", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("out_name", type=str, help="Name of the imputed data file (will be saved at the same location as the original)")
    parser.add_argument("--grouping_columns", nargs="+", help="Columns to use for grouping data.")
    parser.add_argument("--ignore_columns", nargs="+", help="Columns to ignore during imputation.")
    parser.add_argument("--methods", type=str, choices=["LINEAR", "KNN", "RF", "MICE"], default=["LINEAR"], help="Imputation methods to evaluate.")
    parser.add_argument("--save_model_dir", type=str, help="Directory where models should be saved.")
    parser.add_argument("--load_imputed_model", type=str, help="Path to the pre-trained imputation model that should be loaded.")

    args = parser.parse_args()

    out_path = Path(args.data_path).parent / f"{args.out_name}{'.csv' if not args.out_name.endswith('.csv') else ''}"
    assert os.path.exists(Path(args.data_path).parent), "Error, out path does not exist like this!"

    assert Path(args.data_path).stem != Path(args.out_name).stem, "You are trying to overwrite the original data file with"\
    " the imputed one. Aborting imputation, please change out_name!"
    # Load data
    data = pd.read_csv(args.data_path, index_col=0)

    # Verify it does contain NA values
    assert data.isnull().values.any(), "The data you are trying to impute does not contain NANs! Stopping imputation."
    assert not data[args.grouping_columns].isnull().values.any(), "Error, the grouping column contains NAN values! Please make sure you"\
    " drop row with NAN in the grouping column or change the grouping column!"

    impute_columns  = list(
            column for column in data if 
                (not args.grouping_columns or column not in args.grouping_columns) and
                (not args.ignore_columns or column not in args.ignore_columns)
    )
    methods = {
        "LINEAR": LinearImputation,
        "RF": RandomForestImputation,
        "KNN": KNNImputation,
        "MICE": RandomForestIterativeImputation
    }
    # Initialize imputation methods
    method = methods[args.methods]
    # group data in case we have grouping columns (groups are individually imputed)
    if args.grouping_columns:
        grouped = data.groupby(args.grouping_columns)
    else:
        # make temporary group (all data is one group)
        if "tmp" in data.columns:
            raise ValueError(
                "One of the added temporary columns already exists. Please make sure 'tmp' is no column in the data."
            )
        data["tmp"] = 1
        grouped = data.groupby("tmp")
        del data["tmp"]

    for group_name, group_df in tqdm(grouped):
        model = method()
        if isinstance(model, Trainable):
            if args.load_imputed_model:
                model.load_model(os.path.join(args.load_imputed_model, f"{get_model_name(args.methods, group_name)}.joblib"))
            else:
                model.fit(group_df[impute_columns])
                if args.save_model_dir:
                    model.save_model(args.save_model_dir, f"{get_model_name(args.methods, group_name)}.joblib", True)

        imputed = model.impute(group_df[impute_columns])
        data.loc[group_df.index, impute_columns] = imputed
    # sanity check
    assert not data.isnull().values.any(), "Error, data still contains NA values!"
    data.to_csv(out_path)
    print(f"saved imputed file to {out_path}")


if __name__ == "__main__":
    main()

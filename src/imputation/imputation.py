"""Module for imputation. Imputation was heavily inspired by: 
https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html#sphx-glr-auto-examples-impute-plot-iterative-imputer-variants-comparison-py"""
from enum import Enum, auto
from abc import ABC, abstractmethod
from logging import getLogger
import joblib
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


logger = getLogger(__name__)


# Enum representing the methods for imputation
class ImputationLevel(Enum):
    """Level of the imputation. 
    Global: method consider all values.
    Grouped: method imputes subgroups of the data.
    """
    GLOBAL = auto()
    GROUPED = auto()


class ImputationMethod(ABC):
    """Abstract base class representing a imputation method"""
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def get_input_shape(self):
        raise NotImplementedError

    @abstractmethod
    def get_output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def impute(self, data) -> np.ndarray:
        raise NotImplementedError


class Trainable(ABC):
    """Interface representing a trainable entity.
    Examples: Neural Network, Imputation Method, ..."""
    @abstractmethod
    def fit(self, training_data: pd.DataFrame):
        """Fits the training data to the underlying model.

        Args:
            training_data (pd.DataFrame): Data as DataFrame

        Raises:
            NotImplementedError: Abstract method, has to be overwritten.
        """
        raise NotImplementedError
    
    def save_model(self, save_model_dir, model_name, exist_ok=True):
        if save_model_dir:
            os.makedirs(save_model_dir, exist_ok=exist_ok)
            model_path = os.path.join(save_model_dir, model_name)
            self._save(model_path)
            print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        if model_path and os.path.exists(model_path):
            self._load(model_path + (".joblib" if not model_path.endswith(".joblib") else ""))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

    def _save(self, model_path):
        raise NotImplementedError

    def _load(self, model_path):
        raise NotImplementedError


class ImputationEvaluation:
    """Class that can evaluate imputation methods."""
    def __init__(
            self,
            test_data: pd.DataFrame,
            grouping_columns=None,
            ignore_columns=None,
            test_drop_amount=.05,
            tested_sequence_lengths=[1],
        ) -> None:
        # passed parameters
        self.test_data = test_data
        # sequence lengths have to be integer lists
        self.tested_sequence_lengths = tested_sequence_lengths
        assert isinstance(self.tested_sequence_lengths, list)
        assert all(isinstance(x, int) for x in self.tested_sequence_lengths)
        # keep the missing_data between 0 and 1
        self.test_drop_amount = min(max(test_drop_amount, 0.0), 1.0)
        if self.test_drop_amount != test_drop_amount:
            logger.warning(
                "The test_drop_amount was changed since it was set out of bounds: %f -> %f",
                test_drop_amount,
                self.test_drop_amount
            )
        if grouping_columns and not set(grouping_columns).issubset(set(test_data.columns)):
            raise ValueError(f"At least one of the given grouping columns ({grouping_columns})"\
                             f"does not exist in the data frame ({self.test_data.columns})")
        if ignore_columns and not set(ignore_columns).issubset(set(test_data.columns)):
            raise ValueError(f"At least one of the given grouping columns ({ignore_columns})"\
                             f"does not exist in the data frame ({self.test_data.columns})")
        self.ignore_columns = ignore_columns
        self.grouping_columns = grouping_columns

        self.evaluation_scores = ["MAE", "MSE"]
        self.constructed_models = []
        self.results = pd.DataFrame()
        self.groupwise_results = {}

    def get_impute_columns(self):
        impute_columns  = list(
            column for column in self.test_data.columns if 
                (not self.grouping_columns or column not in self.grouping_columns) and
                (not self.ignore_columns or column not in self.ignore_columns)
        )
        return impute_columns

    def drop_values(
            self, group_df: pd.DataFrame, sequence_length: int
        ) -> tuple[pd.DataFrame, dict[tuple[int, int], float]]:
        """Method to drop values in a data frame. The original values that
        are dropped are returned in a dictionary with index as key (row, column)
        and the original value as the values for this key. Additionally, the created
        data frame with the dropped values is returned.

        Returns:
            tuple[pd.DataFrame, dict[tuple[int, int], float]]: Data Frame with dropped values,
            dict with row, column key of dropped values, as well as drop values for this cell
            as value of the dict.
        """
        # columns for imputation (exclude the grouping columns)
        impute_columns  = self.get_impute_columns()
        # the cells dropped is an upper bound, since we do not sample without replacement!
        cells_dropped = int(len(group_df)*len(impute_columns)* self.test_drop_amount)
        drops_per_column = int((cells_dropped / len(impute_columns)) / sequence_length)
        if drops_per_column == 0:
            logger.warning("CAUTION, drops per columns are 0, please increase the percentage of dropped values.")
        # we generate starting points for the sequences that have at least one point between them
        # we can generate sequences for each column
        row_drop_indices = np.empty(shape=(0,), dtype=np.int32)
        column_drop_indices = np.empty(shape=(0,), dtype=np.int32)
        for _ in impute_columns:
            # choices in sequence length steps, also last value should be small enough
            # to not get index errors
            choices = np.arange(len(group_df)-(sequence_length+2), step=sequence_length+1)
            row_drop_indices = np.concatenate([
                row_drop_indices,
                np.random.choice(choices, replace=False, size=drops_per_column)
            ])
            column_drop_indices = np.concatenate([
                column_drop_indices,
                np.random.choice(
                    [group_df.columns.get_loc(col) for col in impute_columns],
                    size=drops_per_column,
                    replace=True
                )
            ])

        # save and replace the values with NAN
        original_values = {}
        df_dropped = group_df.copy()
        for i_start_row, i_col in zip(row_drop_indices, column_drop_indices):
            if (i_start_row, i_col) in original_values.keys():
                continue
            # drop sequences
            for i_row in range(i_start_row, i_start_row+sequence_length):
                original_values[(i_row, i_col)] = self.test_data.iat[i_row, i_col]
                df_dropped.iat[i_row, i_col] = np.nan
        logger.info(
            "Dropped %d values from data frame (%d)",
            df_dropped.isna().sum(), df_dropped.isna().sum()/(len(df_dropped)*len(impute_columns))
        )
        return df_dropped, original_values

    def initialize_results(self, models, sequence_lengths):
        # make data frame to store results
        values_dict = {"model": [model.name for model in models]}
        # setup results DataFrame
        for s in sequence_lengths:
            values_dict.update({
                f"MAE Seq={s}": [np.nan for _ in models],
                f"MSE Seq={s}": [np.nan for _ in models],
                f"MAE Std Seq={s}": [np.nan for _ in models],
                f"MSE Std Seq={s}": [np.nan for _ in models],
                f"N imputed Seq={s}": [np.nan for _ in models]
            })
        df_results = pd.DataFrame(values_dict)
        df_results = df_results.set_index("model")
        return df_results

    def process_group(self, group_df, original_group_values, models, impute_columns):
        group_mae_list = {model.name: [] for model in models}
        group_mse_list = {model.name: [] for model in models}
        group_n_imputed_list = {model.name: [] for model in models}

        for i, model in enumerate(models):
            model_clone = model.__class__()
            if isinstance(model_clone, Trainable):
                model_clone.fit(self.test_data[impute_columns])

            df_imputed = group_df.copy()
            df_imputed[impute_columns] = model_clone.impute(group_df[impute_columns])
            df_imputed = df_imputed.values

            y = np.array(list(original_group_values.values()))
            y_imputed = np.array([df_imputed[coord] for coord in original_group_values.keys()])
            
            # Sanity check
            assert np.isnan(y).sum() == 0, "Error, some of the original values contain NaN values."
            assert np.isnan(y_imputed).sum() == 0, "Error, some of the imputed values contain NaN values."

            mae = np.abs(y - y_imputed).mean()
            mse = ((y - y_imputed) ** 2).mean()

            group_mae_list[model.name].append(mae)
            group_mse_list[model.name].append(mse)
            group_n_imputed_list[model.name].append(len(original_group_values.keys()))
        
        return group_mae_list, group_mse_list, group_n_imputed_list


    def update_results(self, df_results, models, sequence_length, group_mae_list, group_mse_list, group_n_imputed_list):
        for model in models:
            model_name = model.name
            df_results.loc[model_name, f"MAE Seq={sequence_length}"] = np.mean(group_mae_list[model_name])
            df_results.loc[model_name, f"MSE Seq={sequence_length}"] = np.mean(group_mse_list[model_name])
            df_results.loc[model_name, f"MAE Std Seq={sequence_length}"] = np.std(group_mae_list[model_name])
            df_results.loc[model_name, f"MSE Std Seq={sequence_length}"] = np.std(group_mse_list[model_name])
            df_results.loc[model_name, f"N imputed Seq={sequence_length}"] = np.mean(group_n_imputed_list[model_name])


    def test_models(self, models: list[ImputationMethod]) -> None:
        """Method to test multiple models on the same training data. Imputation testing is done by
        dropping random data and training the models on this new data frame. Then, we compare the
        imputed values for the dropped values with the original ones to get a view on how well the
        methods worked.

        Args:
            models (list[ImputationMethod]): A list of all models that should be checked
        """
        # DataFrame to store results
        df_results = self.initialize_results(models, self.tested_sequence_lengths)
        # Columns that are actually imputed
        impute_columns = self.get_impute_columns()
        group_scores = []

        # group data in case we have grouping columns (groups are individually imputed)
        if self.grouping_columns:
            grouped = self.test_data.groupby(self.grouping_columns)
        else:
            # make temporary group (all data is one group)
            if "tmp" in self.test_data.columns:
                raise ValueError(
                    "One of the added temporary columns already exists. Please make sure 'tmp' is no column in the data."
                )
            self.test_data["tmp"] = 1
            grouped = self.test_data.groupby("tmp")
            del self.test_data["tmp"]

        total_iterations = len(self.tested_sequence_lengths) * grouped.ngroups * len(models)
        progress_bar = tqdm(total=total_iterations, desc="Testing Models")

        for sequence_length in self.tested_sequence_lengths:

            group_mae_list = {model.name: [] for model in models}
            group_mse_list = {model.name: [] for model in models}
            group_n_imputed_list = {model.name: [] for model in models}

            for group_name, group_df in grouped:
                df_dropped, original_values = self.drop_values(group_df, sequence_length)
                group_mae, group_mse, group_n_imputed = self.process_group(
                    df_dropped,
                    original_values,
                    models,
                    impute_columns
                )
                progress_bar.update(len(models))

                for model_name in group_mae.keys():
                    group_mae_list[model_name].extend(group_mae[model_name])
                    group_mse_list[model_name].extend(group_mse[model_name])
                    group_n_imputed_list[model_name].extend(group_n_imputed[model_name])
                # update group scores
                for model_name in group_mae.keys():
                    for i in range(len(group_mae[model_name])):
                        group_scores.append({
                            'group_name': group_name,
                            'model_name': model_name,
                            'seq_len': sequence_length,
                            'mae': group_mae[model_name][i],
                            'mse': group_mse[model_name][i],
                            'n_imputed': group_n_imputed[model_name][i]
                        })

            self.update_results(
                df_results,
                models,
                sequence_length,
                group_mae_list,
                group_mse_list,
                group_n_imputed_list
            )
            progress_bar.close()
        
        self.groupwise_results = df_results = pd.DataFrame(group_scores)
        self.results = df_results

    def get_groupwise_results(self) -> dict:
        """Returns the evaluation scores per group, per model from the
        last run.

        Returns:
            dict: Results as dict, keys are the group names (1 if no
            grouping column was given), contains a dict for each model
            that contains all evaluation scores as key, value pairs.
        """
        return self.groupwise_results

    def get_results(self) -> pd.DataFrame:
        """Returns the results (as data frame) of the last test run

        Returns:
            pd.DataFrame: Results, indices are the model names, 
            the columns are metrics, cells are the resulting values for the model
            for the metric
        """
        return self.results

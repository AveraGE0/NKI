"""Module for a (framework) specific dataset"""
import pandas as pd
import numpy as np


def split_train_val(df, id_col, seq_col, val_ratio=0.2):
    train_dfs = []
    val_dfs = []

    for _, group in df.groupby(id_col):
        group = group.sort_values(seq_col)
        n_train_samples = max(1, round(len(group) * (1 - val_ratio)))
        train_df = group.iloc[:n_train_samples]
        val_df = group.iloc[n_train_samples:]
        train_dfs.append(train_df)
        val_dfs.append(val_df)

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    return train_df, val_df




def load_data(
    train_path: str,
    test_path: str,
    ignore_cols: list[str],
    id_col: str,
    seq_col: str,
    categorical_cols: list[str]) -> tuple[pd.DataFrame]:
    """Loads the dataset, splits it, removes unnamed and ignore
    columns and converts categorical columns.

    Args:
        train_path (str): path to training data (csv)
        test_path (str): path to test data (csv)
        ignore_cols (list[str]): columns that will be dropped for training and evaluation
        id_col (str): column with ids (for splitting)
        seq_col (str): column that indicates the order of rows (for splitting)
        categorical_cols (list[str]): categorical columns (these will be converted to string!)

    Returns:
        tuple[pd.DataFrame]: train, validation and test DataFrames
    """
    # Load datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Create validation from test set
    train_data, val_data = split_train_val(
        train_data,
        id_col=id_col,
        seq_col=seq_col,
        val_ratio=0.2
    )

    train_ignore = train_data[ignore_cols]
    val_ignore = val_data[ignore_cols]
    test_ignore = test_data[ignore_cols]
    # Drop unused or unwanted (unnamed) columns!
    train_data = train_data.drop(columns=ignore_cols)
    val_data = val_data.drop(columns=ignore_cols)
    test_data = test_data.drop(columns=ignore_cols)

    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    val_data = val_data.loc[:, ~val_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]

    # convert categorical columns
    for col in categorical_cols:
        train_data[col] = train_data[col].astype(str)
        val_data[col] = val_data[col].astype(str)
        test_data[col] = test_data[col].astype(str)

    return train_data, val_data, test_data, train_ignore, val_ignore, test_ignore


def split_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str
) -> tuple[pd.DataFrame]:
    """Function to split of the target and actual data.
    Returns tuples with (X, y) data for each dataset.

    Args:
        train_data (pd.DataFrame): train df.
        val_data (pd.DataFrame): validation df.
        test_data (pd.DataFrame): test_df.
        target_col (str): target column (will be split off).

    Returns:
        tuple[pd.DataFrame]: (X, y) for train, validation and test set
        in that exact order.
    """
    # Split into features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]

    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]

    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class ModelDataset:
    """Represents a dataset used for modeling. We require three isolated sets,
    train, validation adn test set, provided as indices. This class ensures that
    - indices in specific groups are unique
    - no duplicates are contained between groups
    - if indicted, unnamed columns will be removed
    The main purpose is to ensure a safe and mistake free loading of the corresponding
    datasets, using this more abstract class
    """
    def __init__(self, train_indices: np.ndarray, validation_indices: np.ndarray, test_indices: np.ndarray) -> None:
        pass
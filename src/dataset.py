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
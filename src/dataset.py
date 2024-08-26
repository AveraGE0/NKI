"""Module for a (framework) specific dataset"""
import pandas as pd
import numpy as np


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
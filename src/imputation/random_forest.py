"""Module for RandomForest Imputation"""
from src.imputation.imputation import ImputationMethod, Trainable
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import pandas as pd
import joblib


class RandomForestImputation(ImputationMethod, Trainable):
    def __init__(self, name="RandomForest"):
        super().__init__(name)
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_jobs=-1),
            max_iter=10,
            random_state=42
        )

    def get_input_shape(self):
        return None

    def get_output_shape(self):
        return None

    def fit(self, training_data: pd.DataFrame):
        self.imputer.fit(training_data)

    def impute(self, data):
        return self.imputer.transform(data)

    def _save(self, model_path):
        joblib.dump(self.imputer, model_path, compress=3)

    def _load(self, model_path):
        self.imputer = joblib.load(model_path)

"""Module for MICE imputation"""
from src.imputation.imputation import ImputationMethod, Trainable
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
import pandas as pd
import joblib


class RandomForestIterativeImputation(ImputationMethod, Trainable):
    """Imputation method imitation MICE using RandomForest as the base model"""
    def __init__(self, name="RandomForestIterative"):
        super().__init__(name)
        self.imputer = IterativeImputer(
            random_state=0,
            estimator=RandomForestRegressor(
                #n_estimators=4,
                #max_depth=10,
                bootstrap=True,
                #max_samples=0.5,
                n_jobs=-1,
                #random_state=0,
            ),
            max_iter=1000,
            tol = 1e-4
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
"""Module for KNN imputation"""
from src.imputation.imputation import ImputationMethod
from sklearn.impute import KNNImputer


class KNNImputation(ImputationMethod):
    def __init__(self, name="KNN_Imputer") -> None:
        super().__init__(name)
        self.regressor = KNNImputer(n_neighbors=2, weights="uniform")

    def get_input_shape(self):
        return None

    def get_output_shape(self):
        return None

    def impute(self, data):
        return self.regressor.fit_transform(data)

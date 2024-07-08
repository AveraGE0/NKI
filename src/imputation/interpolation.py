"""Module for interpolation methods for imputation"""
from src.imputation.imputation import ImputationMethod
import numpy as np


class LinearImputation(ImputationMethod):
    """Class the implements linear imputation for time series sequences."""
    def __init__(self, name="LinearImputation") -> None:
        super().__init__(name)
    def get_input_shape(self):
        pass

    def get_output_shape(self):
        pass

    def linear_impute(self, values: np.ndarray):
        """Method to impute all np.NaN values using linear interpolation.
        Values in the beginning and end of the given sequence are imputed using mean
        imputation.

        Args:
            values (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        # safe start value of NaN sequence
        nan_start = None
    
        for i, current_value in enumerate(values):
            if np.isnan(current_value):
                if nan_start is None:
                    nan_start = i
                # sequence goes to last value or is only the last value, use mean imputation
                if i == len(values)-1:
                    values[nan_start:] = np.nanmean(values)
            # we have reached the end of a NAN sequence: a beginning and a non nan value afterwards
            elif nan_start is not None:
                # if no first value is found we use mean imputation
                if nan_start == 0:
                    imputed = np.nanmean(values)
                # otherwise do linear
                else:
                    imputed = np.linspace(
                        start=values[nan_start-1],
                        stop=current_value,
                        num=i-nan_start+2
                    )[1:-1]  # exclude both start and stop value
                values[nan_start:i] = imputed
                # reset sequence start
                nan_start = None
        # sanity check, we do not want any NaNs left
        assert np.isnan(values).sum() == 0,\
        f"Error, linear imputation still had {np.isnan(values).sum()} nan values left"
        return values

    def impute(self, data):
        # with linear, impute each column separately
        for column in data:
            data.loc[:, column] = self.linear_impute(data[column].values)
        return data.values
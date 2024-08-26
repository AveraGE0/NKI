import pandas as pd
import numpy as np
from src.imputation import LinearImputation


def test_single_middle_nan():
    # Test case 1: Single NaN in the middle
    values = np.array([1, 2, np.nan, 4, 5])
    expected = np.array([1, 2, 3, 4, 5])
    df_middle_nan = pd.DataFrame({"A": values})
    
    model = LinearImputation()
    df_imputed = model.impute(df_middle_nan)
    
    np.testing.assert_array_almost_equal(df_imputed, np.expand_dims(expected, axis=-1))

def test_single_leading_nan():
    # Test case 2: Single NaN at the start
    values = np.array([np.nan, 2, 3, 4, 5])
    expected = np.array([3.5, 2, 3, 4, 5])
    df_leading_nan = pd.DataFrame({"A": values})
    model = LinearImputation()
    df_imputed = model.impute(df_leading_nan.copy())
    
    np.testing.assert_array_almost_equal(df_imputed, np.expand_dims(expected, axis=-1))


def test_single_trailing_nan():
    # Test case 3: Single NaN at the end
    values = np.array([1, 2, 3, 4, np.nan])
    expected = np.array([1, 2, 3, 4, 2.5])
    df_trailing_nan = pd.DataFrame({"A": values})
    model = LinearImputation()
    df_imputed = model.impute(df_trailing_nan.copy())
    
    np.testing.assert_array_almost_equal(df_imputed, np.expand_dims(expected, axis=-1))


def test_any_nan_long():
    # Test case: Large array with random NaNs
    values = np.random.rand(1000)
    mask = np.random.choice([1, 0], size=values.shape, p=[.1, .9]).astype(bool)
    values[mask] = np.nan
    df_leading_nans = pd.DataFrame({"A": values})
    
    model = LinearImputation()
    df_imputed = model.impute(df_leading_nans.copy())
    
    assert not np.isnan(df_imputed).any()  # Ensure no NaNs left


def test_multiple_leading_nans():
    df_leading_nans = pd.DataFrame({"A": [np.NaN, np.NaN, np.NaN, 1.0, 2.0, 3.0], "B": [np.NaN, np.NaN, 30.0, 1.0, 2.0, 3.0]})
    model = LinearImputation()
    df_imputed = model.impute(df_leading_nans.copy())
    assert np.isnan(df_imputed).sum() == 0
    assert (df_imputed[:, 0] == [2.0, 2.0, 2.0, 1.0, 2.0, 3.0]).all()
    assert (df_imputed[:, 1] == [9.0, 9.0, 30.0, 1.0, 2.0, 3.0]).all()

def test_multiple_trailing_nans():
    df_trailing_nans = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.NaN, np.NaN, np.NaN], "B": [30.0, 1.0, 2.0, 3.0, np.NaN, np.NaN]})
    model = LinearImputation()
    df_imputed = model.impute(df_trailing_nans.copy())
    assert np.isnan(df_imputed).sum() == 0
    assert (df_imputed[:, 0] == [1.0, 2.0, 3.0, 2.0, 2.0, 2.0]).all()
    assert (df_imputed[:, 1] == [30.0, 1.0, 2.0, 3.0, 9.0, 9.0]).all()


def test_multiple_between_nans():
    df_between_nans = pd. DataFrame({"A": [1.0, np.NaN, np.NaN, np.NaN, 2.0, 3.0], "B": [30.0, 1.0, np.NaN, np.NaN, 4.0, 3.0]})
    model = LinearImputation()
    df_imputed = model.impute(df_between_nans.copy())
    assert np.isnan(df_imputed).sum() == 0
    assert (df_imputed[:, 0] == [1.0, 1.25, 1.5, 1.75, 2.0, 3.0]).all()
    assert (df_imputed[:, 1] == [30.0, 1.0, 2.0, 3.0, 4.0, 3.0]).all()
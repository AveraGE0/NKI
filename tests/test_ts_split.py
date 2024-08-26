import pandas as pd
import pytest
from src.time_series_splitting import train_test_date_split


def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Received empty data frame!"):
        train_test_date_split(df, 0.7, 'date')

def test_invalid_train_test_ratio():
    df = pd.DataFrame({'date': pd.date_range(start='2023-01-01', periods=10)})
    with pytest.raises(ValueError, match="Train test ratio out of range"):
        train_test_date_split(df, 1.2, 'date')
    with pytest.raises(ValueError, match="Train test ratio out of range"):
        train_test_date_split(df, -0.1, 'date')

def test_missing_date_column():
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    with pytest.raises(ValueError, match="The given date column 'date' was not in the given data frame!"):
        train_test_date_split(df, 0.7, 'date')

def test_missing_grouping_columns():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'date': pd.date_range(start='2023-01-01', periods=3),
        'value': [10, 20, 30]
    })
    with pytest.raises(ValueError, match="At least one of the given grouping columns does not exist"):
        train_test_date_split(df, 0.7, 'date', grouping_columns=['nonexistent'])

def test_split_without_grouping():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=6),
        'value': [10, 20, 30, 40, 50, 60]
    })
    train_df, test_df = train_test_date_split(df, 0.5, 'date')
    assert len(train_df) == 3
    assert len(test_df) == 3
    assert all(train_df['date'] < test_df['date'].min())

def test_split_with_grouping():
    df = pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'date': pd.date_range(start='2023-01-01', periods=6),
        'value': [10, 20, 30, 40, 50, 60]
    })
    train_df, test_df = train_test_date_split(df, 0.5, 'date', grouping_columns=['id'])
    assert len(train_df) == 3
    assert len(test_df) == 3
    for group in df['id'].unique():
        assert len(train_df[train_df['id'] == group]) <= len(df[df['id'] == group]) / 2
        assert len(test_df[test_df['id'] == group]) <= len(df[df['id'] == group]) / 2

def test_split_with_different_ratios():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'value': range(10)
    })
    train_df, test_df = train_test_date_split(df, 0.8, 'date')
    assert len(train_df) == 8
    assert len(test_df) == 2
    train_df, test_df = train_test_date_split(df, 0.3, 'date')
    assert len(train_df) == 3
    assert len(test_df) == 7
    assert all(train_df['date'] < test_df['date'].min())

if __name__ == "__main__":
    pytest.main()

"""Module to split time series into meaningful train and test sections"""
from pathlib import Path
import logging
#from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


logger = logging.getLogger(__name__)


def train_test_date_split(
        df_data: pd.DataFrame,
        train_test_ratio: float,
        date_column: str,
        grouping_columns: list[str]=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data according to the date. Also, grouping is possible if the grouping
    column(s) is/are given. If a grouping column is given, the data frame is split into groups
    according to the unique values of this column. The subgroups are then individually split into
    train/test. For example, we could split with the participant id as a grouping factor, so each
    participant would be split (still does not lead to data leakage).

    Args:
        df_data (pd.DataFrame): data being split
        train_test_ratio (float): ratio of train to test, must be in [0, 1]
        date_column (str): the column in which we have the date (needed for sorting)
        grouping_columns (list, optional): Columns used for grouping. If given, the data
        will be grouped according to the unique values in the given column. For each group,
        the split will be done independently. For example, this can be used to have each
        patient in the train and test data by splitting per patient with the given ratio.
        This only impacts how the data will be split, the outputs will still be two
        DataFrames. Defaults to None.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train, test data as new DataFrames, split by date
    """
    # sanity checks first
    if df_data.empty:
        raise ValueError("Received empty data frame!")
    if not 0 < train_test_ratio < 1:
        raise ValueError(
            f"Train test ratio out of range: {train_test_ratio} - expected between (0, 1)"
        )
    if date_column not in df_data.columns:
        raise ValueError(f"The given date column '{date_column}' was not in the given data frame!")
    if grouping_columns and not set(grouping_columns).issubset(set(df_data.columns)):
        raise ValueError(
            "At least one of the given grouping columns does not exist! "\
            f"{grouping_columns} not in {df_data.columns}"
        )

    df_data = df_data.sort_values(by=date_column)
    # make index groups for the splits
    train_frames = []
    test_frames = []

    if grouping_columns:
        grouped = df_data.groupby(grouping_columns)
        for _, group in grouped:
            split_index = int(len(group) * train_test_ratio)
            train_frames.append(group.iloc[:split_index])
            test_frames.append(group.iloc[split_index:])
    else:
        split_index = int(len(df_data) * train_test_ratio)
        train_frames.append(df_data.iloc[:split_index])
        test_frames.append(df_data.iloc[split_index:])

    df_train = pd.concat(train_frames).reset_index(drop=True)
    df_test = pd.concat(test_frames).reset_index(drop=True)

    return df_train, df_test


if __name__ == '__main__':
    test_data = pd.read_csv("./data/example_data.csv", index_col=0)
    data_train, data_test = train_test_date_split(
        test_data, 0.8, date_column="date", grouping_columns=["id"]
    )

    out_dir = Path("./data")
    data_train.to_csv(out_dir / "train_data.csv")
    data_test.to_csv(out_dir / "test_data.csv")

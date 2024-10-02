"""Module to test the validation splitting function"""
import pandas as pd
from src.dataset import split_train_val


def test_validation_split_order():
    """Function to test weather the data is split
    according to the right order."""

    df_train = pd.DataFrame({
        "id":      [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "week":    [1, 3, 2, 2, 2, 3, 3, 1, 1],
        "feature": [1, 5, 5, 2, 4, 6, 3, 3, 4]
    })
    df_train, df_val = split_train_val(df_train, id_col="id", seq_col="week", val_ratio=0.4)

    # right length
    assert len(df_train) == 6 and len(df_val) == 3
    # right order
    id_1_features_train = df_train[df_train["id"] == 1].loc[:, "feature"]
    id_2_features_train = df_train[df_train["id"] == 2].loc[:, "feature"]
    id_3_features_train = df_train[df_train["id"] == 3].loc[:, "feature"]
    assert (id_1_features_train == [1, 2]).all(), f"{id_1_features_train.values} != [1, 2]"
    assert (id_2_features_train == [3, 4]).all(), f"{id_2_features_train.values} != [3, 4]"
    assert (id_3_features_train == [4, 5]).all(), f"{id_3_features_train} != [4, 5]"

    assert (df_val[df_val["id"] == 1].loc[:, "feature"] == [3]).all()
    assert (df_val[df_val["id"] == 2].loc[:, "feature"] == [5]).all()
    assert (df_val[df_val["id"] == 3].loc[:, "feature"] == [6]).all()


def test_validation_split_low_data():
    """Function to test the behavior if insufficient test data
    is available"""

    df_train = pd.DataFrame({
        "id": [1],
        "week": [1],
        "feature": [1]
    })

    df_train, df_val = split_train_val(df_train, "id", "feature")
    assert len(df_train) == 1
    assert len(df_val) == 0, "Caution, splitting favors validation set which might"\
    "lead to for individuals being only present in validation but not test!"

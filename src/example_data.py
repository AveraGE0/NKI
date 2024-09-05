"""Module to create fake data. This is to test any methods, since real
patient data on private machines is not allowed..."""
import random
import pandas as pd
import numpy as np


def generate_random_dates(start_date, end_date, n):
    start_u = start_date.value // 10**9
    end_u = end_date.value // 10**9
    # normalize will set the hour, minutes and seconds to 0
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s').normalize()


def create_example_corsano_data(n_samples, noise_magnitude=0.1, missing_data=0.0, outcome="regression") -> pd.DataFrame:
    """Function to generate fake data. The data will have a pattern (increasing columns have
    +1 on average), but noise is added (if noise magnitude > 0), which slightly randomizes the
    data. Additionally, to make the data more close to the original, missing data (per rate,
    randomly) can be added.

    Args:
        n_samples (_type_): Amount of samples generated
        noise_magnitude (float, optional): Noise Magnitude. No noise if it is set to 0, normal
        random noise if set to 1. Defaults to 0.1.
        missing_data (float, optional): Percentage of missing data. 1.0 Means the dataset will
        only contain NAs, 0 means there will not be
        any missing values. Defaults to 0.0.

    Returns:
        pd.DataFrame: DataFrame. Contains id, date and a number of other (random noise) columns.
    """
    missing_data = min(max(missing_data, 0.0), 1.0)  # keep the missing_data between 0 and 1

    # Create ID column
    df_data = pd.DataFrame()
    df_data['pat_id'] = pd.Series(
        random.choices([f"1100{'0' if v<10 else ''}{v}" for v in range(3)], k=n_samples),
        dtype=str
    )

    # Create date column
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-01-01')

    df_data['date'] = generate_random_dates(start_date, end_date, len(df_data))

    # Create numeric columns
    columns  = ["respiration_rate", "bpm", "temp_sk1"]
    fake_data = np.zeros(shape=(n_samples, len(columns)))
    for i in range(fake_data.shape[1]):
        fake_data[:,i] = i
    # add gaussian noise
    noise = noise_magnitude * np.random.randn(n_samples, len(columns))
    fake_data = fake_data + noise
    df_data = pd.concat([df_data, pd.DataFrame(fake_data, columns=columns)], axis=1)

    # add outcome
    if outcome == "regression":
        df_data["outcome"] = 1*df_data["respiration_rate"] + 2*df_data["bpm"] + 3*df_data["temp_sk1"] + noise_magnitude * np.random.randn(n_samples)
    elif outcome == "classification":
        df_data["outcome"] = np.random.randint(0, 3, size=(n_samples,))
    else:
        raise ValueError("Invalid outcome type received!")
    # note that this is an upper bound, since we might have multiple cells twice in this list
    # since the actual amount of dropped values is not too important (just testing),
    # I will accept this
    cells_dropped = int(n_samples*len(columns)*missing_data)
    row_drop_indices = np.random.randint(low=0, high=n_samples, size=cells_dropped)
    column_drop_indices = np.random.randint(low=2, high=len(columns)+2, size=cells_dropped)
    for i_row, i_col in zip(row_drop_indices, column_drop_indices):
        df_data.iat[i_row, i_col] = np.nan
    return df_data


if __name__ == "__main__":
    # regression (not missing)
    df_fake = create_example_corsano_data(10000)
    df_fake.to_csv("./data/example_data_regression.csv")
    
    # classification (not missing)
    df_fake = create_example_corsano_data(10000, outcome="classification")
    df_fake.to_csv("./data/example_data_classification.csv")
    
    # with missing
    #df_fake = create_example_corsano_data(10000, missing_data=0.5)
    print(
        "Actual NAs: "\
        f"{sum(df_fake.isnull().sum())} "\
        f"({sum(df_fake.isnull().sum())/(len(df_fake)*len(df_fake.columns))}%)"
    )
    #df_fake.to_csv("./data/example_imputation_data_test.csv")

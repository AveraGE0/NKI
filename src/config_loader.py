import os
import configparser
import json
# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the configuration values
train_path_regression = config["paths"]["train_regression"]
test_path_regression = config["paths"]["test_regression"]

categorical_columns_regression = config["data_regression"]["categorical_columns"]
if not categorical_columns_regression:
    categorical_columns_regression = []
else:
    categorical_columns_regression = categorical_columns_regression.split(', ')

ignore_columns_regression = config["data_regression"]["ignore_columns"]
if not ignore_columns_regression:
    ignore_columns_regression = []
else:
    ignore_columns_regression = ignore_columns_regression.split(', ')

target_column_regression = config["data_regression"]["target_column"]

# classification variables
train_path_classification = config["paths"]["train_classification"]
test_path_classification = config["paths"]["test_classification"]

categorical_columns_classification = config["data_classification"]["categorical_columns"]
if not categorical_columns_classification:
    categorical_columns_classification = []
else:
    categorical_columns_classification = categorical_columns_classification.split(', ')

ignore_columns_classification = config["data_classification"]["ignore_columns"]
if not ignore_columns_classification:
    ignore_columns_classification = []
else:
    ignore_columns_classification = ignore_columns_classification.split(', ')

target_column_classification = config["data_classification"]["target_column"]

# Load configuration from JSON file
with open('feature_types.json', 'r', encoding="utf-8") as config_file:
    feature_types = json.load(config_file)


def get_feature_types(columns: list[str]):
    return [feature_types['feature_types'][col] for col in columns]
# Extract feature types from the configuration
#if not os.path.isfile(train_path):
#    raise ValueError(f"The train path specified in the config is invalid: {train_path}")
#if not os.path.isfile(test_path):
#    raise ValueError(f"The test path specified in the config is invalid: {test_path}")

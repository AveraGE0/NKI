import os
import configparser
import json
# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the configuration values
train_path = config["paths"]["train"]
test_path = config["paths"]["test"]

categorical_columns = config["data"]["categorical_columns"]
if not categorical_columns:
    categorical_columns = []

ignore_columns = config["data"]["ignore_columns"]
if not ignore_columns:
    ignore_columns = []

target_column = config["data"]["target_column"]

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

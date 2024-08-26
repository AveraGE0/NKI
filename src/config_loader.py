import os
import configparser

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

#if not os.path.isfile(train_path):
#    raise ValueError(f"The train path specified in the config is invalid: {train_path}")
#if not os.path.isfile(test_path):
#    raise ValueError(f"The test path specified in the config is invalid: {test_path}")

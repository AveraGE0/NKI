"""Module responsible for creating, managing and accessing models saved to the file structure"""
import os
from datetime import datetime
import pandas as pd


def unique_model_name(model_type: str) -> str:
    return f"model_{model_type}_{datetime.now().strftime(format='%d%m%Y_%H%M%S')}"


def create_model_directory(model_name: str, model_description: dict) -> None:
    """Creates the file structure for a new model. Note that both the name of the model and the
    description of the model is mandatory. We only want to accept models where at least some
    basic properties are described (what type of features, what score is predicted, etc.).
    This is forced to keep track of models that perform different tasks.

    Args:
        model_name (str): The unique name of the model. Should be generated with the unique model name
        function (not mandatory though).
        model_description: Dictionary with str: str entries describing the trained model. At least
        the type, features used and score predicted should be contained (this is not enforced)
    """
    # create dir if it does not exist
    create_path = f"./models/{model_name}"
    if not os.path.exists(create_path):
        os.makedirs(create_path)
    else:
        raise OSError(f"The model directory you are trying to create does already exist: {create_path}")
    # create description file
    with open(f"{create_path}/model_description.md", mode="+w", encoding="utf-8") as description_file:
        description_file.write(f"# {model_name}\n\n")
        for key, value in model_description.items():
            description_file.write(f"## {key}\n\n")
            description_file.write(f"{value}\n\n")
    return


def get_model_performance_list(model_type: str, metric_file_names: list[str]) -> pd.DataFrame:
    """Function to receive a DataFrame with name and score
    of all available models

    Args:
        model_type (str): if given, only substrings containing this name will be considered. Leave empty for all models
        metric_file_names (list[str]): list of all file names from which metrics are loaded. Make sure that all models
        contain all of those files.

    Returns:
        pd.DataFrame: containing all models of the given type along with all metrics of the given metric files.
    """
    test = os.listdir("./models")
    model_dirs = [directory for directory in os.listdir("./models") if os.path.isdir(f"./models/{directory}")]
    df_performance = pd.DataFrame()
    for model_name in model_dirs:
        model_path = f"./models/{model_name}"
        model_data = {"index": model_name}
        # load files with evaluation data
        for metric_file in metric_file_names:
            # careful, this creates lists as types
            model_data.update(pd.read_csv(f"{model_path}/{metric_file}").to_dict(orient="list"))

        df_performance = pd.concat([df_performance, pd.DataFrame().from_dict(model_data)])
    return df_performance



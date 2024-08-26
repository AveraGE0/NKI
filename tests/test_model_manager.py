import src.modeling.model_manager as manager
import os
import pandas as pd


def test_model_name_creation():
    name = manager.unique_model_name(model_type="test")
    assert "test" in name and len(name) > len("model") + len("test") + len("00000000_000000")


def test_model_dir_creation():
    name = "unittest_model"
    expected_path = f"./models/{name}"
    assert not os.path.isdir(expected_path)
    manager.create_model_directory(name, {"features": "None, this is a test model", "outcome": "also None..."})
    assert os.path.isdir(expected_path)
    assert os.path.isfile(f"{expected_path}/model_description.md")
    # cleanup
    os.remove(f"{expected_path}/model_description.md")
    os.rmdir(expected_path)


def test_model_performance_list():
    # create two fake models with two performance csv files
    for index, name in enumerate(["unittest_model1", "unittest_model2"]):
        manager.create_model_directory(name, {})
        path = f"./models/{name}"
        pd.DataFrame().from_dict({"mse": [index], "mae": [index]}).to_csv(f"{path}/test_metrics1.csv")
        pd.DataFrame().from_dict({"vmse": [index+1]}).to_csv(f"{path}/test_metrics2.csv")
    df_performance = manager.get_model_performance_list(model_type="", metric_file_names=["test_metrics1.csv", "test_metrics2.csv"])
    for metric in ["mse", "mae", "vmse"]:
        assert metric in df_performance.columns
    # cleanup
    for name in ["unittest_model1", "unittest_model2"]:
        os.remove(f"./models/{name}/model_description.md")
        os.remove(f"./models/{name}/test_metrics1.csv")
        os.remove(f"./models/{name}/test_metrics2.csv")
        os.rmdir(f"./models/{name}")

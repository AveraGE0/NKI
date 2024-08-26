from src.imputation import RandomForestImputation
from src.imputation import RandomForestIterativeImputation
from src.imputation import KNNImputation
from src.imputation import LinearImputation
import pandas as pd
from src.imputation.imputation import ImputationEvaluation


def test_imputation_evaluator():
    test_data = pd.read_csv("./data/example_data.csv", index_col=0)
    test_data['date'] = pd.to_datetime(test_data['date'])
    tester = ImputationEvaluation(
        test_data,
        tested_sequence_lengths=[1, 4],
        grouping_columns=["id"],
        ignore_columns=["date"]
    )
    tester.test_models([
        LinearImputation(),
        RandomForestImputation(),
        RandomForestIterativeImputation(),
        KNNImputation()
    ])
    print(tester.get_results())
    print(tester.get_groupwise_results())

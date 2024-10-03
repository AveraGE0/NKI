import pandas as pd
import joblib
import optuna
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
from src.modeling.model_manager import create_model_directory, unique_model_name, add_description
from src.config_loader import train_path_regression as train_path
from src.config_loader import test_path_regression as test_path
from src.config_loader import categorical_columns_regression as categorical_columns
from src.config_loader import ignore_columns_regression as ignore_columns
from src.config_loader import target_column_regression as target_column
from src.config_loader import id_column, seq_column
from src.config_loader import config
from src.evaluation.metric_computation import make_save_metrics
from src.evaluation.plots import plot_feature_importance
from src.dataset import load_data, split_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df_train, df_val, df_test, train_ignore, val_ignore, test_ignore = load_data(
        train_path=train_path,
        test_path=test_path,
        id_col=id_column,
        seq_col=seq_column,
        ignore_cols=ignore_columns,
        categorical_cols=categorical_columns
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        df_train,
        df_val,
        df_test,
        target_col=target_column
    )

    # Create model directory
    model_name = unique_model_name("optimization_ebm")
    create_model_directory(model_name, {
        f"{model_name}": "Optimization run of EBM models with varying parameters using Optuna",
        "Used Features": ", ".join(X_test.columns),
        "Target": target_column,
        "Dataset Name": train_path
    })

    model_path = f'./models/{model_name}'

    def objective(trial):
        """Objective function for Optuna. Returns a specific value for
        a trial. The value is the negative MSE, since we are maximizing
        the outcome.

        Args:
            trial (_type_): trial configuration

        Returns:
            float: Negative MSE (for maximization)
        """
        # Define the hyperparameter search space
        max_rounds = trial.suggest_int('max_rounds', 25000, 35000)  # similar to n_estimators, amount of boosting rounds
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, step=0.01)
        max_leaves = trial.suggest_int('max_leaves', 2, 50)  # complexity of individual boosting iteration
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 100)  # prevents too small splits -> similar to regularization
        max_bins = trial.suggest_int('max_bins', 32, 512)  # binning accuracy (performance <-> overfitting)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.1)  # l1 regularization
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0, step=0.5)  # l2 regularization

        # Create the model with the sampled hyperparameters
        model = ExplainableBoostingRegressor(
            max_rounds=max_rounds,
            learning_rate=learning_rate,
            max_leaves=max_leaves,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            random_state=42,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        return -mean_squared_error(y_val, y_pred)

    # Create an Optuna study and optimize it
    study = optuna.create_study(
        storage=f'sqlite:///{model_path}/optuna_study.db',
        direction='maximize'
    )

    study.optimize(
        objective,
        n_trials=int(config["optuna"]["n_trials"]),
        n_jobs=int(config["optuna"]["n_jobs"])
    )

    # Print the best hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')

    trial = study.best_trial

    trial_str = '  Value: {}\n'.format(trial.value)
    trial_str += '  Params: '
    for key, value in trial.params.items():
        trial_str +='    {}: {}\n'.format(key, value)
    
    print(trial_str)
    
    add_description(
        f'{model_path}/model_description.md',
        headline="Best Trial",
        content=trial_str
    )

    # Train the final model with the best hyperparameters
    best_params = study.best_params
    best_model = ExplainableBoostingRegressor(
        **best_params,
        random_state=42,
        feature_names=X_train.columns.tolist()
    )

    X_final_train = pd.concat([X_train, X_val])
    y_final_train = pd.concat([y_train, y_val])
    ignore_final_train = pd.concat([train_ignore, val_ignore])

    best_model.fit(X_final_train, y_final_train)

    add_description(
        f'{model_path}/model_description.md',
        headline="Model Parameters",
        content='\n'.join(f'{key}: {value}' for key, value in best_model.get_params().items())
    )

    # Save the final model using joblib with compression
    joblib.dump(best_model, f'{model_path}/optimized_ebm_model.pkl', compress=3)

    eval_sets = []
    for name, X, y, ignore in [
        ("train", X_final_train, y_final_train, ignore_final_train),
        ("test", X_test, y_test, test_ignore)
    ]:
        y_pred = best_model.predict(X)
        df_data: pd.DataFrame = X.copy()
        df_data["actual"] = y
        df_data["predicted"] = y_pred
        df_data = df_data.join(ignore)
        eval_sets.append((name, df_data))

    make_save_metrics(model_path, eval_sets, y_col="actual", y_pred_col="predicted")

    fig = plot_feature_importance(best_model)

    # Save the plot to a specific path
    fig.savefig(f'{model_path}/ebm_feature_importances.png')
    plt.close()

"""
Программа: Тренировка данных
Версия: 0.1
"""

import optuna
from optuna import Study
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from ..data.split_dataset import split_train_test
from ..train.metrics import save_metrics


def rmsle_loss(y_true, y_pred):
    """
    Пользовательская целевая функция ошибки RMSLE
    для LightGBM objective
    """
    grad = 2 * (np.log(y_pred + 1) - np.log(y_true + 1)) / (y_pred + 1)
    hess = (-2 * np.log(y_pred + 1) + 2 * np.log(y_true + 1) + 2) / (y_pred + 1) ** 2
    return grad, hess


def rmsle_eval(y_true, y_pred):
    """
    Пользовательская целевую функцию обнаружеия переобучения RMSLE
    для LightGBM eval_metric
    """
    loss = np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))
    return "rmsle", loss, False


def objective_lgbm(
    trial,
    X: pd.DataFrame,
    y: pd.DataFrame,
    N_FOLDS: int,
    random_state: int,
    eval_metric: rmsle_eval,
    objective: rmsle_loss,
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param X: данные объект-признаки
    :param y: данные с целевой переменной
    :param N_FOLDS: кол-во фолдов
    :param random_state: random_state
    :param eval_metric: метрика оценки ранней остановке
    :param objective: задача обучения
    :return: среднее значение метрики по фолдам
    """
    lgbm_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000, step=50),
        # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100),
        # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100),
        # "min_gain_to_split": trial.suggest_int("min_gain_to_split", 0, 20),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        # "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt', 'rf', 'dart']),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "objective": trial.suggest_categorical("objective", [objective]),
    }

    cv = KFold(n_splits=N_FOLDS, shuffle=True)

    cv_predicts = np.empty(N_FOLDS)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmsle")
        model = LGBMRegressor(**lgbm_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=eval_metric,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=0,
        )

        preds = model.predict(X_test)

        try:
            cv_predicts[idx] = np.sqrt(mean_squared_log_error(y_test, preds))
        except ValueError:
            continue
            # pass

    return np.mean(cv_predicts)


def find_optimal_params(
    data: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
    validation_size: float,
    N_FOLD: int,
    n_trial: int,
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data: датасет
    :param target: название целевой переменной
    :param test_size: доля тестовой выборки от общего объема данных
    :param random_state: random state
    :param validation_size: доля валидационной выборки от train объема данных
    :param N_FOLD: количество фолдов
    :param n_trial: количество иттераций
    :return: [LGBMRegressor tuning, Study]
    """

    (
        X_train,
        X_train_,
        X_test,
        X_val,
        y_train,
        y_train_,
        y_test,
        y_val,
    ) = split_train_test(data, target, test_size, random_state, validation_size)

    study = optuna.create_study(direction="minimize", study_name="LGB_optuna")

    func = lambda trial: objective_lgbm(
        trial,
        X_train,
        y_train,
        N_FOLDS=N_FOLD,
        random_state=random_state,
        eval_metric=rmsle_eval,
        objective=rmsle_loss,
    )

    study.optimize(func, n_trials=n_trial, show_progress_bar=True)
    return study


def train_model(
    data: pd.DataFrame,
    study: Study,
    target: str,
    test_size: float,
    random_state: int,
    validation_size: float,
    metric_path: str,
) -> LGBMRegressor:
    """
    Обучение модели на лучших параметрах
    :param data: датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param test_size: доля тестовой выборки от общего объема данных
    :param random_state: random state
    :param validation_size: доля валидационной выборки от train объема данных
    :param metric_path: путь до папки с метриками
    :return: LGBMClassifier
    """

    (
        X_train,
        X_train_,
        X_test,
        X_val,
        y_train,
        y_train_,
        y_test,
        y_val,
    ) = split_train_test(data, target, test_size, random_state, validation_size)

    # тренировка с оптимальными параметрами
    lgbm_optuna = LGBMRegressor(**study.best_params)
    eval_set = [(X_val, y_val)]
    lgbm_optuna.fit(
        X_train_,
        y_train_,
        eval_metric=rmsle_eval,
        eval_set=eval_set,
        verbose=0,
        early_stopping_rounds=100,
    )


    # сохранение метрик
    save_metrics(
        data_x=X_test,
        data_y=y_test,
        delta=1,
        model=lgbm_optuna,
        metric_path=metric_path,
    )
    return lgbm_optuna
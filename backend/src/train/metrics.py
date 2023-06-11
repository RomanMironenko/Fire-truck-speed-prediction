"""
Программа: Получение метрик
Версия: 0.1
"""
import json
import yaml

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
)

import pandas as pd
import numpy as np


def r2_adjusted(y_true: list, y_pred: list, X_test: np.array) -> float:
    """
    Coefficient of determination (R2 adjusted) metric
    Коэффициент детерминации (парная регрессия)
    """
    N_objects = len(y_true)
    N_features = X_test.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (N_objects - 1) / (N_objects - N_features - 1)


def mpe(y_true: list, y_pred: list) -> float:
    """
    The Mean Percentage Error (MPE) metric
    Средняя процентная ошибка
    """
    return np.mean((y_true - y_pred) / y_true)


def mape(y_true: list, y_pred: list) -> float:
    """
    The Mean Absolute Percentage Error (MAPE) metric
    Cредняя абсолютная процентная ошибка
    """
    return np.mean(np.abs((y_pred - y_true) / y_true))


def wape(y_true: list, y_pred: list) -> float:
    """
    The Weighted Absolute Percentage Error (WAPE) metric
    Взвешенная абсолютная процентная ошибка
    """
    return np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)


def huber_loss(y_true, y_pred, delta):
    """
    The Huber loss (Huber_loss) metric
    Функция ошибки Хьюбера
    """
    assert len(y_true) == len(y_pred), "Разные размеры данных"
    huber_sum = 0
    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) <= delta:
            huber_sum += 0.5 * (y_true[i] - y_pred[i]) ** 2
        else:
            huber_sum += delta * (abs(y_true[i] - y_pred[i]) - 0.5 * delta)
    huber_sum /= len(y_true)
    return huber_sum


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    The Root Mean Squared Log Error (RMSLE) metric
    Логаритмическая ошибка средней квадратичной ошибки
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def create_dict_metrics(y_test, y_pred, X_test, delta) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные по y
    :param y_pred: предсказанные значения
    :param X_pred: реальные данные по X
    :param delta: delta для Huber loss
    :return: словарь с метриками
    """
    try:
        rmsle_value =  round(rmsle(y_test, y_pred), 3)
    except:
        pass

    dict_metrics = {
        "MAE": round(round(mean_absolute_error(y_test, y_pred), 3)),
        "MSE": round(round(mean_absolute_error(y_test, y_pred), 3)),
        "RMSE": round(round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)),
        "RMSLE": rmsle_value,
        # "RMSLE": round(rmsle(y_test, y_pred), 3),
        "R2 adjusted": round(r2_adjusted(y_test, y_pred, X_test), 3),
        "Huber_loss": round(huber_loss(y_test, y_pred, delta), 3),
        "MPE_%": round(mpe(y_test, y_pred) * 100, 3),
        "MAPE_%": round(mape(y_test, y_pred) * 100, 3),
        "WAPE_%": round(wape(y_test, y_pred) * 100, 3),
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, delta, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(
        y_test=data_y.values,
        y_pred=model.predict(data_x),
        X_test=data_x,
        delta=delta,
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["backend"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics

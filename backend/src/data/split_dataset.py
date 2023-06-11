"""
Программа: Разделение данных на train/test
Версия: 0.1
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(
    dataset: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
    validation_size: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Разделение данных на train/test
    :param dataset: датасет
    :param target: название целевой переменной
    :param test_size: доля тестовой выборки от общего объема данных
    :param random_state: random state
    :param validation_size: доля валидационной выборки от train объема данных
    :return: train/test датасеты
    """
    X = dataset.drop(target, axis=1)
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=test_size, random_state=random_state
    )

    X_train_, X_val, y_train_, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        shuffle=True,
        random_state=random_state,
    )

    return X_train, X_train_, X_test, X_val, y_train, y_train_, y_test, y_val

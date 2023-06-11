"""
Программа: Frontend получение данных по пути и чтение
Версия: 0.1
"""

import pandas as pd


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_csv(dataset_path)

"""
Программа: Получение предсказания на основе обученной модели
Версия: 0.1
"""

import os
import yaml
import joblib
import pandas as pd


def data_preprocessing(config_path, datalist) -> pd.DataFrame:
    """
    Предобработка входных данных и получение предсказаний
    :param config_path: путь до конфигурационного файла
    :param datalist: данные
    :return: DataFrame
    """
    # Получение параметров
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

        model_auto = config["preprocessing"]["car_characteristics"]
        cols = config["evaluate"]["columns"]

        features = [
            [
                datalist[0],
                datalist[1],
                datalist[2],
                model_auto["max_speed_km/h"][datalist[0]],
                model_auto["full_mass_kg"][datalist[0]],
                model_auto["engine_power_l_s"][datalist[0]],
                model_auto["type_auto"][datalist[0]],
                datalist[3],
            ]
        ]

        data = pd.DataFrame(features, columns=cols)

        for col in data.select_dtypes(object).columns:
            data[col] = data[col].astype("category")

    return data


def pipeline_evaluate(config_path, datalist: list) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param config_path: путь до конфигурационного файла
    :param datalist: данные
    :return: предсказания
    """
    # Получение параметров
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

        train_config = config["backend"]

    data = data_preprocessing(config_path, datalist)

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(data).tolist()

    return prediction

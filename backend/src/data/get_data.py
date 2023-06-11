"""
Программа: Получение данных из файла
Версия: 0.1
"""

import os
import re
from datetime import datetime
import pandas as pd
import gpxpy
import gpxpy.gpx
from transliterate import translit


def identifiers_auto(path_id: str, sep: str, encoding: str) -> pd.DataFrame:
    """
    Загрузка индификаторов автомобилей
    :param path_id: путь до файла с индификаторами авто
    :param sep: разделитель для csv файла
    :param encoding: кодировка файла
    :return: DataFrame
    """
    return pd.read_csv(path_id, sep=sep, encoding=encoding)


def dict_csv_model(
    identifiers_auto_: pd.DataFrame, path_tracks: str, model_auto: str, id_auto: str
) -> dict:
    """
    Создание словаря со списками (Модель:[список CSV треков под определенную модель авто]
    :param identifiers_auto_: DataFrame c индификаторами автомобилей
    :param path_tracks: путь до папки с треками
    :param model_auto: название столбца с моделями авто из identifiers_auto
    :param id_auto: название столбца с id авто из identifiers_auto
    :return: DataFrame
    """

    # Создание словаря со списками (Модель:[список CSV треков под определенную модель авто]
    dict_csv_model = {}

    # Перебор по моделям авто
    for model in identifiers_auto_[model_auto].unique():

        # Генератор списка содержащий список CSV треков под определенную модель авто
        list_csv = [
            filename
            for filename in os.listdir(path_tracks)
            if int(filename.split("_")[0])
            in list(
                identifiers_auto_[identifiers_auto_[model_auto] == model][
                    id_auto
                ].unique()
            )
        ]

        # Добавление списка с словарь с ключом МОДЕЛЬ
        if len(list_csv) != 0:
            dict_csv_model[str(model)] = list_csv.copy()
            list_csv.clear()
        else:
            list_csv.clear()

    return dict_csv_model


def data_df_track(path_tracks: str, dict_csv_model: dict) -> pd.DataFrame:
    """
    Создание DataFrame с данными
    :param path_tracks: путь до папки с треками
    :param dict_csv_model: словарь со списками
    (Модель:[список CSV треков под определенную модель авто]
    :return: DataFrame
    """

    # Создание конечного DataFrame
    data = pd.DataFrame()

    # Перебор моделей авто в цикле
    for key in dict_csv_model.keys():

        # Перебор файлов треков в цикле
        for gpx_csv in dict_csv_model[key]:
            with open(path_tracks + gpx_csv, "r", encoding="utf-8") as gpx_file:
                gpx = gpxpy.parse(gpx_file)

            # Создание списка для словаря для преобразования в DataFrame
            initial_data = []

            # Перебор информации в треке в цикле
            for track in gpx.tracks:
                for segment in track.segments:
                    for count, point in enumerate(segment.points):

                        # Добавление в список словаря с контрольными точками
                        initial_data.append(
                            {
                                "id": gpx_csv.split(".")[0],
                                "model": str(
                                    re.sub(
                                        " ",
                                        "_",
                                        translit(
                                            key, language_code="ru", reversed=True
                                        ),
                                    )
                                ),
                                "progression": count + 1,
                                "latitude": point.latitude,
                                "longitude": point.longitude,
                                "elevation": point.elevation,
                                "date_and_time": datetime.strftime(
                                    point.time, "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
            # Закрытие файла с треком
            gpx_file.close()

            # Преобразование списка со славорем в DataFrame
            initial_data_df = pd.DataFrame(initial_data)

            # Создание DataFrame где будет содержаться информации о первой точки в отрезке
            start = (
                initial_data_df.copy(deep=True)
                .drop(initial_data_df.index[-1], axis=0)
                .reset_index()
                .drop("index", axis=1)
            )
            # Переименовывание столбцов
            start.columns = [
                "id",
                "model",
                "start_progression",
                "start_latitude",
                "start_longitude",
                "start_elevation",
                "start_date_and_time",
            ]

            # Создание DataFrame где будет содержаться информации о последней точки в отрезке
            end = (
                initial_data_df.copy(deep=True)
                .drop(initial_data_df.index[0], axis=0)
                .reset_index()
                .drop(["id", "index", "model"], axis=1)
            )

            # Переименовывание столбцов
            end.columns = [
                "end_progression",
                "end_latitude",
                "end_longitude",
                "end_elevation",
                "end_date_and_time",
            ]

            # Соединение двух DataFrame'ов
            full = pd.concat([start, end], axis=1)

            # Соединение двух DataFrame'ов
            data = pd.concat([data, full], axis=0, ignore_index=True)

    return data


def get_dataset(
    path_id: str,
    sep: str,
    encoding: str,
    path_tracks: str,
    model_auto: str,
    id_auto: str,
) -> pd.DataFrame:
    """
    Получение данных
    Загрузка индификаторов автомобилей
    :param path_id: путь до файла с индификаторами авто
    :param sep: разделитель для csv файла
    :param encoding: кодировка файла
    :param path_tracks: путь до папки с треками
    :param model_auto: название столбца с моделями авто из identifiers_auto
    :param id_auto: название столбца с id авто из identifiers_auto
    :return: DataFrame
    """
    auto = identifiers_auto(path_id, sep=sep, encoding=encoding)

    id_csv_model = dict_csv_model(auto, path_tracks, model_auto, id_auto)

    return data_df_track(path_tracks, id_csv_model)
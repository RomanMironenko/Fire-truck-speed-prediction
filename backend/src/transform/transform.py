"""
Программа: Предобработка данных
Версия: 0.1
"""

import calendar
from datetime import timedelta
import pandas as pd
import requests
from geopy.distance import geodesic


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return: DataFrame
    """
    return data.astype(change_type_columns, errors="raise")


def distance(data: pd.DataFrame, name_columns_lat_lon: list) -> list:
    """
    Добавление расстояния между соседними точками в одном треке в м
    :param data: датасет
    :param name_columns_lat_lon: список с широтой и долготой названия
     столбцов начальной и конечной точками отрезка
    :return: список с расстояниями
    """
    distance = []
    for count in data.index:
        distance.append(
            geodesic(
                (
                    data.loc[count][name_columns_lat_lon[0]],
                    data.loc[count][name_columns_lat_lon[1]],
                ),
                (
                    data.loc[count][name_columns_lat_lon[2]],
                    data.loc[count][name_columns_lat_lon[3]],
                ),
            ).m
        )
    return distance


def travel_time(data: pd.DataFrame, name_columns_time: list) -> list:
    """
    Добавление времени прохождения между соседними точками в одном треке
    :param data: датасет
    :param  name_columns_time: список столбцов с начальным и конечным
     временем точек отрезка формат времени datetime64[ns]
    :return: список с временем
    """
    # Расчет времени движения
    travel_time = []
    for count in data.index:
        travel_time.append(
            (
                data.loc[count][name_columns_time[1]]
                - data.loc[count][name_columns_time[0]]
            ).total_seconds()
        )
    return travel_time


def speed_km_h(
    data: pd.DataFrame, name_columns_distance: str, name_columns_travel_time: str
):
    """
    Добавление скорости
    :param data: датасет
    :param name_columns_distance: название столбца с расстоянием, м
    :param name_columns_travel_time: название столбца с временем, с
    :return: столбец со скоростью км/ч
    """
    return data[name_columns_distance] / data[name_columns_travel_time] * 3600 / 1000


def days_week(data: pd.DataFrame, column_data: str) -> list:
    """
    Добавление дня недели
    :param data: датасет
    :param colum_data: название столбца с датой
    :return: список с днями недели
    """
    return [calendar.day_name[day.weekday()] for day in data[column_data]]


def type_days(data: pd.DataFrame, name_column_time: str) -> object:
    """
    Добавление характеристик автомобиля
    :param data: датасет
    :param name_columns_time: название столбца с датой с типом datetime64[ns]
    :return: датасет с типом дня
    """
    type_days = []
    for day in data[name_column_time]:
        type_days.append(day.date())
    type_days = pd.Series(list(set(type_days))).astype("datetime64[ns]")
    types_days = {}

    # Определение рабочено, не рабочего и пред не рабочего дня
    for day in type_days:
        response = requests.get(
            "https://isdayoff.ru/{}{}{}".format(
                day.year, day.strftime("%m"), day.strftime("%d")
            )
        )
        if response.status_code == 200:
            if response.json() == 0:
                rs = requests.get(
                    "https://isdayoff.ru/{}{}{}".format(
                        (day + timedelta(days=1)).year,
                        (day + timedelta(days=1)).strftime("%m"),
                        (day + timedelta(days=1)).strftime("%d"),
                    )
                )
                if rs.json() == 0:
                    types_days[day.date()] = "working_day"

                else:
                    types_days[day.date()] = "before_non-working_day"
            else:
                types_days[day.date()] = "not_working_day"
        else:
            break
    return (data[name_column_time].apply(lambda x: x.date())).map(types_days)


def car_characteristics(
    data: pd.DataFrame, name_column_model: str, characteristics: dict
) -> list:
    """
    Добавление характеристик автомобиля
    :param data: датасет
    :param name_columns_model: название столбца с моделями авто
    :param characteristics: словарь с характеристиками авто
    :return: список характеристиками
    """
    return (data[name_column_model].apply(lambda x: x)).map(characteristics)


def pickup_hour(data: pd.DataFrame, name_column_time: str) -> list:
    """
    Добавление часа когда происходила запись точки в трек
    :param data: датасет
    :param  name_column_time: название столбца с временем в формат времени datetime64[ns]
    :return: список с часами
    """
    return data[name_column_time].apply(lambda x: x.hour)


def transform(
    data: pd.DataFrame,
    change_type_columns: dict,
    name_columns_lat_lon: list,
    name_columns_time: list,
    name_column_time: str,
    feature_auto: dict,
    name_column_model: str,
    drop_columns: list,
    save_data: str,
) -> pd.DataFrame:
    """
    Трансформирование данных полученных после функции get_data
    :param data: DataFrame после функции get_data
    :param change_type_columns: словарь с признаками и типами данных
    :param name_columns_lat_lon: список с широтой и долготой названия
     столбцов начальной и конечной точками отрезка
    :param  name_columns_time: список столбцов с начальным и конечным
     временем точек отрезка формат времени datetime64[ns]
    :param name_column_time: название столбца с датой с типом datetime64[ns]
    :param feature_auto: словарь с характеристиками авто
    :param name_column_model: название столбца с моделями авто
    :param  drop_columns: список с удаляемыми столбцами
    :param  save_data: сохранение данных
    :return: DataFrame
    """

    data = transform_types(data, change_type_columns)

    data["distance_m"] = distance(data, name_columns_lat_lon)

    data["travel_time_s"] = travel_time(data, name_columns_time)

    data["speed_km/h"] = speed_km_h(data, "distance_m", "travel_time_s")

    data["days_of_the_week"] = days_week(data, name_column_time)

    data["day_type"] = type_days(data, name_column_time)

    for feature in feature_auto.keys():
        data[feature] = car_characteristics(
            data, name_column_model, feature_auto[feature]
        )

    data = data[(data["type_auto"] != "ABG") & (data["type_auto"] != "APT")]

    data["pickup_hour"] = pickup_hour(data, name_column_time)

    # Исключаю записи где изменение расстояния между точками менее 5.7 м
    data = data[data["distance_m"] > 5.7]

    # Исключаю записи где присутсвуют значения превыщающие максимально возможную скорость для каждого типа автомобиля
    data = data[data["speed_km/h"] <= data["max_speed_km/h"]]

    # Исключаю записи где присутсвуют значения 0 км/ч
    data = data[data["speed_km/h"] != 0]

    # Удаление не нужны колонок
    data = data.drop(drop_columns, axis=1)

    # Замена типа данных с object на category
    for col in data.select_dtypes(object).columns:
        data[col] = data[col].astype("category")

    # Сохранение DataFrame в csv
    data.to_csv(save_data, index=False)

    return data
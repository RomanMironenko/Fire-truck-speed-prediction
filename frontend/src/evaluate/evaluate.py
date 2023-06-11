"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 0.1
"""

import pandas as pd
import streamlit as st
import requests
import json


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    unique_data = pd.read_csv(unique_data_path, sep=",", encoding="cp1251")

    # поля для вводы данных, используем уникальные значения
    # модель авто
    model = st.sidebar.selectbox("Model", (sorted(unique_data["model"].unique())))

    # Для сортировки дней недели
    day = {
        "Friday": 4,
        "Monday": 0,
        "Saturday": 5,
        "Sunday": 6,
        "Thursday": 3,
        "Tuesday": 1,
        "Wednesday": 2,
    }
    # День недели
    days_of_the_week = st.sidebar.selectbox(
        "Days of the week",
        (sorted(unique_data["days_of_the_week"].unique(), key=day.get)),
    )
    # Тип дня
    day_type = st.sidebar.selectbox(
        "Day type", (sorted(unique_data["day_type"].unique()))
    )

    # Время выезда
    pickup_hour = st.sidebar.selectbox(
        "Pickup hour", (list(map(int, sorted(unique_data["pickup_hour"].unique()))))
    )

    dict_data = {
        "Model": model,
        "Days_of_the_week": days_of_the_week,
        "Day_type": day_type,
        "Pickup_hour": pickup_hour,
    }

    st.write(
        f"""### Данные о вызове:\n
    1) Model: {dict_data['Model']}
    2) Days of the week: {dict_data['Days_of_the_week']}
    3) Day type: {dict_data['Day_type']}
    4) Pickup hour {dict_data['Pickup_hour']}
    """
    )

    # Предсказание
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output} км/ч")
        st.success("Success!")

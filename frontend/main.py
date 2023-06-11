"""
Программа: Frontend часть проекта
Версия: 0.1
"""

import os
import streamlit as st
import yaml
from src.data.get_data import get_dataset
from src.plotting.charts import displot_bar, boxplot_bar
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input

config_path = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://avatars.mds.yandex.net/i?id=c84c438ebf8ddd097d779feec1bd666b201c2958-8270623-images-thumbs&n=13",
        width=600,
    )

    st.markdown("# Описание проекта")
    st.title("MLOps project:  Fire truck speed prediction 🔥 🚒")
    st.write(
        """
       ."""
    )

    # Наименование колонок
    st.markdown(
        """
        ### Описание полей 
            - Model - Модель пожарного автомобиля
            - Days_of_the_week - День недели (понедельник, вторник и т.п.) 
            - Day_type - Тип дня (Рабочий день, пред нерабочий день, не рабочий день)
            - Pickup_hour: Час выезда
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # загрузить и записать набор данных
    data = get_dataset(dataset_path=config["backend"]["train"])
    st.write(data.head())

    # график с checkbox
    displot = st.sidebar.checkbox("Распределение скорости")
    speed_week = st.sidebar.checkbox("Распределение скорости по дням недели")
    speed_week_auto = st.sidebar.checkbox(
        "Распределение скорости по дням недели с учетом типа автомобиля"
    )
    speed_hour = st.sidebar.checkbox("Распределение скорости по часам внутри дня")
    speed_type_days = st.sidebar.checkbox(
        "Распределение скорости в рабочий, не рабочий, пред не рабочий"
    )
    speed_type_auto = st.sidebar.checkbox("Распределение скорости по типам автомобилей")

    if displot:
        st.pyplot(
            displot_bar(
                data=data,
                col="speed_km/h",
                bins=100,
                title="Распределение скорости",
            )
        )
        st.markdown(
            """
             **Распределение не нормальное**
            """)
    if speed_week:
        st.pyplot(
            boxplot_bar(
                data=data,
                x_data="days_of_the_week",
                y_data="speed_km/h",
                title="Распределение скорости по дням недели",
            )
        )
        st.markdown(
            """
             **Наибольшая средняя скорость достигается в Воскресенье**
            """)
    if speed_week_auto:
        st.pyplot(
            boxplot_bar(
                data=data,
                x_data="days_of_the_week",
                y_data="speed_km/h",
                title="Распределение скорости по дням недели с учетом типа автомобиля",
                hue="type_auto",
            )
        )
        st.markdown(
            """
             **Средние значение скорости зависит от дня недели, но не зависит от типа автомобиля. 
             Средние значение скорости по типам автомобилей лежат в узком диапазоне внутри одного дня недели**
            """)
    if speed_hour:
        st.pyplot(
            boxplot_bar(
                data=data,
                x_data="pickup_hour",
                y_data="speed_km/h",
                title="Распределение скорости по часам внутри дня",
            )
        )
        st.markdown(
            """
             **Средняя скорость до 6 часов возрастает, после 6 ч до 8 ч падает, с 9 ч по 17 ч 
             практически не изменяется, 18 ч падает и до 23 ч проиходит планамерный рост**
            """)
    if speed_type_days:
        st.pyplot(
            boxplot_bar(
                data=data,
                x_data="day_type",
                y_data="speed_km/h",
                title="Распределение скорости в рабочий, не рабочий, пред не рабочий",
            )
        )
        st.markdown(
            """
             **Наибольшая средняя скорость достигается в не рабочий день, 
             наименьшая скорость в пред не рабочий день**
            """)
    if speed_type_auto:
        st.pyplot(
            boxplot_bar(
                data=data,
                x_data="type_auto",
                y_data="speed_km/h",
                title="Распределение скорости по типам автомобилей",
            )
        )
        st.markdown(
            """
             **Наибольшая средняя скорость у автомобилей типа АСА потом у АЦ/АНР, а затем у АЛ/АКП**
            """)


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")

    # получение параметров
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["predict_input"]
    unique_data_path = config["backend"]["train"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["backend"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training": training,
        "Prediction": prediction,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()

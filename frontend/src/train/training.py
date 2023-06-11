"""
Программа: Тренировка модели на backend, отображение метрик
Версия: 0.1
"""

import os
import json
import streamlit as st
import requests
# import joblib
# from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Последние метрики
    if os.path.exists(config["backend"]["metrics_path"]):
        with open(config["backend"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"MAE": 0, "MSE": 0, "RMSE": 0, "RMSLE": 0, "R2 adjusted": 0, "Huber_loss": 0,
                       "MPE_%": 0, "MAPE_%": 0, "WAPE_%": 0}

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)

    new_metrics = output.json()["metrics"]

    # Табличка сравнения метрик
    MAE, MSE, RMSE, RMSLE, R2_adjusted = st.columns(5)
    MAE.metric(
        "MAE",
        new_metrics['MAE'],
        f"{new_metrics['MAE'] - old_metrics['MAE']:.3f}",
    )
    MSE.metric(
        "MSE",
        new_metrics["MSE"],
        f"{new_metrics['MSE'] - old_metrics['MSE']:.3f}",
    )
    RMSE.metric(
        "RMSE",
        new_metrics["RMSE"],
        f"{new_metrics['RMSE'] - old_metrics['RMSE']:.3f}",
    )
    RMSLE.metric(
        "RMSLE",
        new_metrics["RMSLE"],
        f"{new_metrics['RMSLE'] - old_metrics['RMSLE']:.3f}"
    )
    R2_adjusted.metric(
        "R2 adjusted",
        new_metrics["R2 adjusted"],
        f"{new_metrics['R2 adjusted'] - old_metrics['R2 adjusted']:.3f}",
    )
    Huber_loss, MPE_, MAPE_, WAPE_ = st.columns(4)
    Huber_loss.metric(
        "Huber_loss",
        new_metrics["Huber_loss"],
        f"{new_metrics['Huber_loss'] - old_metrics['Huber_loss']:.3f}",
    )
    MPE_.metric(
        "MPE_%",
        new_metrics["MPE_%"],
        f"{new_metrics['MPE_%'] - old_metrics['MPE_%']:.3f}",
    )
    MAPE_.metric(
        "MAPE_%",
        new_metrics["MAPE_%"],
        f"{new_metrics['MAPE_%'] - old_metrics['MAPE_%']:.3f}",
    )
    WAPE_.metric(
        "WAPE_%",
        new_metrics["WAPE_%"],
        f"{new_metrics['WAPE_%'] - old_metrics['WAPE_%']:.3f}",
    )
###ModuleNotFoundError: No module named 'src.train.train'###
    # plot study
    # study = joblib.load(os.path.join(config["backend"]["study_path"]))
    # fig_imp = plot_param_importances(study)
    # fig_history = plot_optimization_history(study)
    # st.plotly_chart(fig_imp, use_container_width=True)
    # st.plotly_chart(fig_history, use_container_width=True)
###ModuleNotFoundError: No module named 'src.train.train'###




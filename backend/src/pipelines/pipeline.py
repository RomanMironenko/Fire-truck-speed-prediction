"""
Программа: Сборный конвейер для тренировки модели
Версия: 0.1
"""


import os
import joblib
import yaml

from ..data.get_data import get_dataset
from ..transform.transform import transform
from ..train.train import find_optimal_params, train_model


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param config_path: путь до файла с конфигурациями
    :return: None
    """
    # Получение параметров
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["traning"]
    backend_config = config["backend"]
    evaluate_config = config["evaluate"]

    # получение данных
    get_dataset_data = get_dataset(
        path_id=preprocessing_config["id_auto"],
        sep=preprocessing_config["sep"],
        encoding=preprocessing_config["encoding"],
        path_tracks=preprocessing_config["path_open"],
        model_auto="МОДЕЛЬ",
        id_auto=preprocessing_config["id"],
    )

    # обработка
    transform_data = transform(
        data=get_dataset_data,
        change_type_columns=preprocessing_config["change_type_columns"],
        name_columns_lat_lon=preprocessing_config["distance"],
        name_columns_time=preprocessing_config["travel"],
        name_column_time=preprocessing_config["name_column_time"],
        feature_auto=preprocessing_config["car_characteristics"],
        name_column_model=preprocessing_config["name_column_model"],
        drop_columns=evaluate_config["drop_columns"],
        save_data=backend_config["train"],
    )

    # поиск оптимальных параметров
    study = find_optimal_params(
        data=transform_data,
        target=train_config["target_column"],
        test_size=train_config["test_size"],
        random_state=train_config["random_state"],
        validation_size=train_config["validation_size"],
        N_FOLD=train_config["n_folds"],
        n_trial=train_config["n_trials"],
    )

    # тренировка с оптимальными параметрами
    model = train_model(
        data=transform_data,
        study=study,
        target=train_config["target_column"],
        test_size=train_config["test_size"],
        random_state=train_config["random_state"],
        validation_size=train_config["validation_size"],
        metric_path=backend_config["metrics_path"],
    )

    # сохранение результатов (study, model)
    joblib.dump(model, os.path.join(backend_config["model_path"]))
    joblib.dump(study, os.path.join(backend_config["study_path"]))

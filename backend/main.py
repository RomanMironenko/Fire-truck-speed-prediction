"""
Программа: Модель для прогнозирования скорости движения пожарного автомобиля
Версия: 0.1
"""


import warnings
import optuna

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
config_path = "../config/params.yml"


class Signs(BaseModel):
    """
    Признаки для получения результатов модели
    """

    Model: str
    Days_of_the_week: str
    Day_type: str
    Pickup_hour: int


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=config_path)
    metrics = load_metrics(config_path=config_path)

    return {"metrics": metrics}


@app.post("/predict_input")
def prediction_input(customer: Signs):
    """
    Предсказание модели по введенным данным
    """
    features = [
        customer.Model,
        customer.Days_of_the_week,
        customer.Day_type,
        customer.Pickup_hour,
    ]
    predictions = pipeline_evaluate(config_path=config_path, datalist=features)[0]
    return round(predictions)


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)

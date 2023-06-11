"""
Программа: Отрисовка графиков
Версия: 0.1
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def displot_bar(
    data: pd.DataFrame, col: str, bins: int, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика displot
    :param data: датасет
    :param col: признак для анализа
    :param bins: количество разбиений
    :param title: название графика
    :return: поле рисунка
    """
    sns.set_style("whitegrid")

    ax = sns.displot(data[col], bins=bins)
    plt.title(title, fontsize=20)
    return ax


def boxplot_bar(
    data: pd.DataFrame, x_data: str, y_data: str, title: str, hue=None
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param x_data: признак для анализа
    :param y_data: признак для анализа
    :param hue: группирвока по признаку,
    :param title: название графика
    :return: поле рисунка
    """
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(15, 7))

    sns.boxplot(x=x_data, y=y_data, data=data, hue=hue)
    plt.title(title, fontsize=20)
    plt.ylabel(y_data, fontsize=14)
    plt.xlabel(x_data, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig

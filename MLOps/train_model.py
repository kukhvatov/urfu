from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import mlflow
import pandas as pd
from mlflow.models import infer_signature
import joblib
import warnings

warnings.filterwarnings("ignore")


def split_data(dataframe):
    """Разделение данных на обучающие и валидационные"""
    features = dataframe.drop('popularity', axis=1)
    target = dataframe['popularity']
    return train_test_split(features, target, test_size=0.3, random_state=42)


def evaluate_performance(true_values, predictions):
    """Расчет метрик качества модели"""
    return {
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'r2': r2_score(true_values, predictions)
    }


def train_model():
    search_params = {
        "penalty": ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1.0],
        'solver': ['saga', 'liblinear']
    }

    mlflow.set_experiment("Spotify Track Popularity Analysis LR")
    song_data = pd.read_csv("./cleaned_data.csv")

    with mlflow.start_run():
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(estimator, search_params, cv=5, scoring='r2')

        X_train, X_val, y_train, y_val = split_data(song_data)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_val)

        metrics = evaluate_performance(y_val, predictions)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "popularity_predictor", signature=signature)

        joblib.dump(best_model, "spotify_popularity_model.pkl")
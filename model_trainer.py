from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import explained_variance_score, max_error
import mlflow
import pandas as pd
import joblib
import numpy as np

# Параметры для RandomizedSearch
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

def prepare_features(data):
    """Подготовка признаков и целевой переменной"""
    features = data.drop('popularity', axis=1)
    target = data['popularity']
    return features, target

def evaluate_model(true, pred):
    """Расчет метрик качества"""
    return {
        'mae': np.mean(np.abs(true - pred)),
        'rmse': np.sqrt(np.mean((true - pred)**2)),
        'explained_variance': explained_variance_score(true, pred),
        'max_error': max_error(true, pred)
    }

mlflow.set_experiment("Spotify Popularity Prediction")
data = pd.read_csv("./cleaned_spotify_data.csv")

with mlflow.start_run():
    # Инициализация модели
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Настройка валидации
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Поиск гиперпараметров
    search = RandomizedSearchCV(
        rf, 
        params, 
        cv=tscv,
        n_iter=10,
        scoring='neg_mean_squared_error'
    )
    
    X, y = prepare_features(data)
    search.fit(X, y)
    
    best_model = search.best_estimator_
    predictions = best_model.predict(X)
    
    metrics = evaluate_model(y, predictions)
    
    # Логирование параметров
    mlflow.log_params(search.best_params_)
    
    # Логирование метрик
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)
    
    # Сохранение модели
    joblib.dump(best_model, "best_rf_model.pkl")
    mlflow.sklearn.log_model(best_model, "random_forest_model")

# Получение пути к лучшей модели
runs = mlflow.search_runs()
best_run = runs.loc[runs['metrics.rmse'].idxmin()]
model_path = best_run['artifact_uri'].replace("file://", "") + "/random_forest_model"
print(f"Best model path: {model_path}")

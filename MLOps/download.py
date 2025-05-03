import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from model_training import train_model


def clean_and_transform():
    raw_data = pd.read_csv('dataset.csv')

    # Удаление ненужных данных
    raw_data = raw_data.dropna()
    cols_to_drop = ['track_id', 'track_name', 'album_name']
    processed_data = raw_data.drop(columns=cols_to_drop)

    # Преобразование типов данных
    processed_data['explicit'] = processed_data['explicit'].astype(np.uint8)

    # Кодирование категорий
    categorical_features = ['artists', 'track_genre']
    encoder = OrdinalEncoder()
    for feature in categorical_features:
        processed_data[feature] = encoder.fit_transform(processed_data[[feature]])

    # Удаление коррелирующих признаков
    redundant_features = ['loudness', 'energy', 'danceability',
                          'valence', 'acousticness', 'instrumentalness']
    final_data = processed_data.drop(columns=redundant_features)

    final_data.to_csv('cleaned_data.csv', index=False)
    return True


spotify_dag = DAG(
    dag_id="spotify_training_pipeline",
    start_date=datetime(2025, 5, 1),
    schedule_interval="@weekly",
    max_active_runs=1,
    catchup=False
)

data_cleaning_task = PythonOperator(
    task_id="data_preparation",
    python_callable=clean_and_transform,
    dag=spotify_dag
)

model_training_task = PythonOperator(
    task_id="model_training",
    python_callable=train_model,
    dag=spotify_dag
)

data_cleaning_task >> model_training_task
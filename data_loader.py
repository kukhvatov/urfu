import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import requests
from io import StringIO

def fetch_spotify_data():
    """Загрузка данных через прямой URL с Kaggle"""
    url = "https://www.kaggle.com/api/v1/datasets/download/maharshipandya/-spotify-tracks-dataset"
    response = requests.get(url)
    csv_data = StringIO(response.content.decode('utf-8'))
    raw_df = pd.read_csv(csv_data)
    raw_df.to_csv('spotify_data.csv', index=False)
    return raw_df

def clean_dataset(data_path):
    """Предобработка данных"""
    df = pd.read_csv(data_path)
    
    # Удаление ненужных колонок
    df.drop(columns=['id', 'album_id', 'time_signature'], axis=1, inplace=True)
    
    # Обработка категориальных признаков
    categorical_features = ['artists', 'track_genre']
    transformer = ColumnTransformer(
        [('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features],
        remainder='passthrough'
    )
    
    transformed_data = transformer.fit_transform(df)
    feature_names = transformer.get_feature_names_out()
    processed_df = pd.DataFrame(transformed_data, columns=feature_names)
    
    # Нормализация числовых признаков
    numeric_cols = ['popularity', 'duration_ms', 'key', 'mode', 'tempo']
    processed_df[numeric_cols] = (processed_df[numeric_cols] - processed_df[numeric_cols].mean()) / processed_df[numeric_cols].std()
    
    # Удаление коррелирующих признаков
    processed_df.drop(columns=['loudness', 'energy', 'speechiness'], axis=1, inplace=True)
    
    processed_df.to_csv('cleaned_spotify_data.csv', index=False)
    return True

if __name__ == "__main__":
    raw_data = fetch_spotify_data()
    clean_dataset('spotify_data.csv')

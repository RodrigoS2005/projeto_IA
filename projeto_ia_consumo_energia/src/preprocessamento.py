
import pandas as pd

def criar_features(df):
    df['data'] = pd.to_datetime(df['data'])
    df['dia_semana'] = df['data'].dt.dayofweek
    for lag in [1,3,7]:
        df[f'lag_{lag}'] = df['consumo'].shift(lag)
    return df.dropna()

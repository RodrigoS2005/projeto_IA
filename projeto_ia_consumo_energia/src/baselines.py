import numpy as np
import pandas as pd

def baseline_media(y_train, y_test):
    """Preve sempre a média histórica do treino."""
    media_historica = y_train.mean()
    return np.full(len(y_test), media_historica)

def baseline_dia_anterior(X_test):
    """Preve que o consumo de amanhã será igual ao de ontem (Lag 1)."""
    # Assume que X_test já tem a feature 'lag_1'
    return X_test['lag_1'].values
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calcular_metricas(y_true, y_pred):
    """Retorna MAE, RMSE e MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Evitar divisão por zero no MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return mae, rmse, mape
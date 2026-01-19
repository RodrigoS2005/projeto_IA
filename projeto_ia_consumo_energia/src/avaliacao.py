
from sklearn.metrics import mean_absolute_error

def avaliar_modelo(y_true, y_pred):
    print("MAE:", mean_absolute_error(y_true, y_pred))

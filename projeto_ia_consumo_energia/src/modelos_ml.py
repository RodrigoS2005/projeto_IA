
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def treinar_modelos(X_train, X_test, y_train):
    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo.predict(X_test)

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def obter_modelos():
    """Retorna um dicionário com as instâncias dos modelos configuradas."""
    return {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    }

def treinar_e_prever(modelo, X_train, y_train, X_test):
    """Função utilitária para fit e predict."""
    modelo.fit(X_train, y_train)
    return modelo.predict(X_test)
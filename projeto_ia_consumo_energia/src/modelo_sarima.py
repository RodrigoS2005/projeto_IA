from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def prever_sarima(y_train, exog_train, exog_test, steps):
    """
    Ajusta e preve com SARIMAX.
    Nota: Treina apenas nos dados mais recentes para eficiência.
    """
    # Usamos apenas os últimos 700 dias para treino devido ao custo computacional
    janela_treino = 700
    if len(y_train) > janela_treino:
        y_train_subset = y_train.iloc[-janela_treino:]
        exog_train_subset = exog_train.iloc[-janela_treino:]
    else:
        y_train_subset = y_train
        exog_train_subset = exog_train

    try:
        # Ordem (1,1,1) e Sazonal (0,1,1,7) para capturar o padrão semanal
        model = SARIMAX(y_train_subset, 
                       exog=exog_train_subset,
                       order=(1, 1, 1), 
                       seasonal_order=(0, 1, 1, 7),
                       enforce_stationarity=False, 
                       enforce_invertibility=False)
        
        fit = model.fit(disp=False)
        forecast = fit.predict(start=len(y_train_subset), 
                               end=len(y_train_subset) + steps - 1, 
                               exog=exog_test)
        return forecast
    except Exception as e:
        print(f"Erro no SARIMA: {e}")
        return np.zeros(steps)
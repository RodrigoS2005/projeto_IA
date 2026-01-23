import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

# Importar módulos locais
sys.path.append('src')
from preprocessamento import gerar_dados_simulados, criar_features
from baselines import baseline_media, baseline_dia_anterior
from modelos_ml import obter_modelos, treinar_e_prever
from modelo_sarima import prever_sarima
from avaliacao import calcular_metricas

# --- Configuração ---
pd.options.mode.chained_assignment = None

def main():
    print("=== INÍCIO DO PIPELINE DE PREVISÃO DE ENERGIA ===\n")

    # 1. GERAÇÃO DE DADOS (SIMULAÇÃO)
    print("[1/5] Gerando dados sintéticos via Simulador Smart Grid...")
    df_raw = gerar_dados_simulados(n_registos=15000)
    df = criar_features(df_raw)
    print(f"      Dataset gerado com {len(df)} registos.")

    # 2. SEPARAÇÃO TEMPORAL (SPLIT)
    # Os últimos 365 dias servem como teste (simulando o ano seguinte)
    test_size = 365
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    target = 'consumo'
    
    # --- DEFINIÇÃO DE FEATURES ---
    # CENÁRIO REAL: O modelo só vê a 'temp_previsao' (que tem erro/ruído)
    features_reais = ['mes', 'dia_semana', 'fim_de_semana', 
                      'lag_1', 'lag_3', 'lag_7', 
                      'rolling_mean_7', 'rolling_std_7', 
                      'temp_previsao'] # <--- Com Ruído
    
    # CENÁRIO ORÁCULO: O modelo vê a 'temp_real' (Informação perfeita)
    features_oracle = ['mes', 'dia_semana', 'fim_de_semana', 
                       'lag_1', 'lag_3', 'lag_7', 
                       'rolling_mean_7', 'rolling_std_7', 
                       'temp_real'] # <--- Sem Ruído (Perfeito)

    # Preparar X e y
    X_train = train[features_reais]
    y_train = train[target]
    X_test = test[features_reais]
    y_test = test[target]

    # Preparar X para o Oráculo
    X_train_oracle = train[features_oracle]
    X_test_oracle = test[features_oracle]

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_oracle = StandardScaler()
    X_train_oracle_scaled = scaler_oracle.fit_transform(X_train_oracle)
    X_test_oracle_scaled = scaler_oracle.transform(X_test_oracle)

    resultados_lista = []

    # 3. BASELINES
    print("[2/5] Executando Baselines...")
    # Baseline 1: Média
    y_pred_media = baseline_media(y_train, y_test)
    mae, rmse, mape = calcular_metricas(y_test, y_pred_media)
    resultados_lista.append(['Baseline Média', mae, rmse, mape])

    # Baseline 2: Persistência (Lag 1)
    y_pred_persistencia = baseline_dia_anterior(X_test)
    mae, rmse, mape = calcular_metricas(y_test, y_pred_persistencia)
    resultados_lista.append(['Baseline Dia Anterior', mae, rmse, mape])

    # 4. MODELOS DE ML (CENÁRIO REAL - Com erro na meteorologia)
    print("[3/5] Treinando modelos ML (Cenário Real)...")
    modelos = obter_modelos()
    
    # Ridge
    y_pred_ridge = treinar_e_prever(modelos['Ridge'], X_train_scaled, y_train, X_test_scaled)
    mae, rmse, mape = calcular_metricas(y_test, y_pred_ridge)
    resultados_lista.append(['Ridge Regression', mae, rmse, mape])

    # Random Forest (Não precisa de scale, mas funciona igual)
    y_pred_rf = treinar_e_prever(modelos['RandomForest'], X_train, y_train, X_test)
    mae, rmse, mape = calcular_metricas(y_test, y_pred_rf)
    resultados_lista.append(['Random Forest', mae, rmse, mape])

    # Gradient Boosting
    y_pred_gb = treinar_e_prever(modelos['GradientBoosting'], X_train, y_train, X_test)
    mae, rmse, mape = calcular_metricas(y_test, y_pred_gb)
    resultados_lista.append(['Gradient Boosting', mae, rmse, mape])

    # 5. ORÁCULO (LIMIT SUPERIOR - Cenário Perfeito)
    print("[4/5] Calculando Oráculo (Limite Teórico)...")
    # Usamos o Ridge como base para o Oráculo, mas alimentamos com DADOS PERFEITOS
    oracle_model = obter_modelos()['Ridge'] # Reutiliza instância nova se quiser, ou a mesma
    oracle_model.fit(X_train_oracle_scaled, y_train)
    y_pred_oracle = oracle_model.predict(X_test_oracle_scaled)
    
    mae, rmse, mape = calcular_metricas(y_test, y_pred_oracle)
    resultados_lista.append(['Oráculo (Info Perfeita)', mae, rmse, mape])

    # 6. SARIMA (Estatístico)
    print("[5/5] Ajustando SARIMA (pode demorar)...")
    exog_train_sarima = train[['temp_previsao', 'fim_de_semana']]
    exog_test_sarima = test[['temp_previsao', 'fim_de_semana']]
    
    y_pred_sarima = prever_sarima(y_train, exog_train_sarima, exog_test_sarima, steps=len(y_test))
    # SARIMA pode retornar NaN ou zero se falhar, lidamos com isso no avaliacao
    mae, rmse, mape = calcular_metricas(y_test, y_pred_sarima)
    resultados_lista.append(['SARIMAX', mae, rmse, mape])

    # --- APRESENTAÇÃO FINAL ---
    df_resultados = pd.DataFrame(resultados_lista, columns=['Modelo', 'MAE', 'RMSE', 'MAPE (%)'])
    print("\n" + "="*50)
    print("RESULTADOS FINAIS DO EXPERIMENTO")
    print("="*50)
    print(df_resultados)
    print("="*50)
    
    # Exportar para CSV para por no relatório se necessário
    df_resultados.to_csv('resultados_finais.csv', index=False)
    print("Resultados exportados para 'resultados_finais.csv'.")

    # Gráfico simples
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:100], label='Real', color='black', linewidth=2)
    plt.plot(y_pred_gb[:100], label='Gradient Boosting (Real)', linestyle='--')
    plt.plot(y_pred_oracle[:100], label='Oráculo (Perfeito)', linestyle=':', color='green', linewidth=2)
    plt.title("Comparação: Modelo Real vs Oráculo (Primeiros 100 dias)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
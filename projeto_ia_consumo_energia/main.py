import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# --- Configurações Globais ---
warnings.filterwarnings('ignore')
sns.set_style("whitegrid") # Estilo visual mais limpo para gráficos
plt.rcParams.update({'figure.max_open_warning': 0})
np.random.seed(42) # Reprodutibilidade

# ==============================================================================
# 1. GERAÇÃO DE DADOS (Simulação Avançada)
# ==============================================================================
def gerar_dados_complexos(n_registos=15000):
    """
    Gera um dataset com 15.000 registos mimetizando dados reais de energia:
    - Sazonalidade Anual (Inverno/Verão) e Semanal (Dias úteis vs Fim de semana).
    - Relação não-linear com temperatura (U-Shape).
    - Outliers (Eventos extremos).
    """
    datas = pd.date_range(start='1983-01-01', periods=n_registos, freq='D')
    df = pd.DataFrame({'data': datas})
    
    # Variável auxiliar temporal
    t = np.arange(n_registos)
    
    # --- Clima Simulado ---
    # Temperatura segue uma onda senoidal anual + ruído diário
    temp_sazonal = 10 * np.sin(2 * np.pi * t / 365)
    df['temp_media'] = 15 + temp_sazonal + np.random.normal(0, 3, n_registos)
    df['temp_min'] = df['temp_media'] - np.random.uniform(3, 7, n_registos)
    df['temp_max'] = df['temp_media'] + np.random.uniform(3, 7, n_registos)
    # Humidade inversa à temperatura (simplificação) + ruído
    df['humidade'] = 60 + 10 * np.cos(2 * np.pi * t / 365) + np.random.normal(0, 8, n_registos)
    df['humidade'] = df['humidade'].clip(10, 100)

    # --- Consumo Elétrico (Target) ---
    # 1. Base constante
    base = 200
    # 2. Efeito Temperatura (Não linear / U-Shape): Gasta-se aquecimento (<15) e AC (>22)
    efeito_temp = 1.5 * (df['temp_media'] - 18)**2 
    # 3. Efeito Semanal: Fim de semana consome menos (Indústria parada)
    dia_semana = df['data'].dt.dayofweek
    efeito_fds = np.where(dia_semana >= 5, -50, 20)
    # 4. Tendência de longo prazo (eletrificação)
    tendencia = t * 0.005
    
    # Consumo Final = Soma dos componentes + Ruído aleatório
    df['consumo'] = base + efeito_temp + efeito_fds + tendencia + np.random.normal(0, 15, n_registos)
    
    # 5. Injeção de Outliers (Eventos raros/extremos)
    # Escolhemos aleatoriamente 0.5% dos dias para serem picos de consumo
    idx_outliers = np.random.choice(n_registos, size=int(n_registos*0.005), replace=False)
    df.loc[idx_outliers, 'consumo'] += np.random.uniform(100, 200, len(idx_outliers))
    
    df.set_index('data', inplace=True)
    return df

# ==============================================================================
# 2. ENGENHARIA DE FEATURES (Feature Engineering)
# ==============================================================================
def criar_features(df_input):
    df = df_input.copy()
    
    # Variáveis Temporais
    df['mes'] = df.index.month
    df['dia_semana'] = df.index.dayofweek
    df['fim_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    # Lags (O que aconteceu há X dias?)
    # Nota: Lag 1 é frequentemente o preditor mais forte (Autocorrelação)
    for lag in [1, 3, 7]:
        df[f'lag_{lag}'] = df['consumo'].shift(lag)
        
    # Estatísticas Móveis (Rolling Window)
    # CUIDADO: Usar shift(1) para evitar Data Leakage (usar dados de hoje para prever hoje)
    df['rolling_mean_7'] = df['consumo'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['consumo'].shift(1).rolling(window=7).std()
    
    # Remover linhas com NaN geradas pelos lags
    df.dropna(inplace=True)
    return df

# ==============================================================================
# 3. AVALIAÇÃO
# ==============================================================================
def calcular_metricas(y_true, y_pred, modelo_nome):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'Modelo': modelo_nome, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

# ==============================================================================
# 4. PIPELINE PRINCIPAL
# ==============================================================================

print("--- 1. Preparação dos Dados ---")
df_raw = gerar_dados_complexos()
df = criar_features(df_raw)

# Definição de Variáveis
target = 'consumo'
features = [col for col in df.columns if col != target]

# Separação Temporal (Holdout sem baralhar)
# Últimos 365 dias para teste (Simulando o "futuro")
test_size = 365
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Normalização (Fit no treino, Transform no teste)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # O teste usa a escala do treino!

resultados = []

print("\n--- 2. Execução de Baselines ---")
# Baseline 1: Média Histórica
y_pred_mean = np.full(len(y_test), y_train.mean())
resultados.append(calcular_metricas(y_test, y_pred_mean, "Baseline: Média"))

# Baseline 2: Persistência (Naive) - O valor de amanhã será igual ao de ontem
y_pred_persistencia = X_test['lag_1']
resultados.append(calcular_metricas(y_test, y_pred_persistencia, "Baseline: Persistência"))

print("\n--- 3. Treino de Modelos ML ---")

# A. Ridge Regression (Linear Regularizada)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
resultados.append(calcular_metricas(y_test, y_pred_ridge, "Ridge Regression"))

# B. Random Forest (Ensemble Bagging)
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
resultados.append(calcular_metricas(y_test, y_pred_rf, "Random Forest"))

# C. Gradient Boosting (Ensemble Boosting) - Geralmente o "Estado da Arte" tabular
gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
resultados.append(calcular_metricas(y_test, y_pred_gb, "Gradient Boosting"))

print("\n--- 4. Modelo Estatístico (SARIMA) ---")
# Nota: Treinar SARIMA em 15.000 pontos é computacionalmente inviável para uma demo rápida.
# Vamos usar os últimos 700 dias (~2 anos) do treino para ajustar o SARIMA.
sarima_train_y = y_train.iloc[-700:]
sarima_train_exog = X_train[['temp_media', 'fim_de_semana']].iloc[-700:]
sarima_test_exog = X_test[['temp_media', 'fim_de_semana']]

print("   > Ajustando SARIMA (pode demorar um pouco)...")
try:
    # Ordem (1,1,1) e Sazonal (0,1,1,7) para capturar padrão semanal
    model_sarima = SARIMAX(sarima_train_y, 
                           exog=sarima_train_exog,
                           order=(1, 1, 1), 
                           seasonal_order=(0, 1, 1, 7),
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
    sarima_fit = model_sarima.fit(disp=False)
    y_pred_sarima = sarima_fit.predict(start=len(sarima_train_y), 
                                       end=len(sarima_train_y)+len(y_test)-1, 
                                       exog=sarima_test_exog)
    # Realign index
    y_pred_sarima.index = y_test.index
    resultados.append(calcular_metricas(y_test, y_pred_sarima, "SARIMAX"))
except Exception as e:
    print(f"   > Falha no SARIMA: {e}")

# ==============================================================================
# 5. APRESENTAÇÃO DE RESULTADOS E GRÁFICOS
# ==============================================================================

# Tabela de Resultados
df_res = pd.DataFrame(resultados).set_index('Modelo')
print("\n=== RESULTADOS FINAIS ===")
print(df_res.sort_values('RMSE'))

# --- Gráfico 1: Previsão Temporal (Zoom) ---
plt.figure(figsize=(15, 6))
zoom = 60 # Visualizar apenas os primeiros 60 dias do teste para clareza
plt.plot(y_test.index[:zoom], y_test.values[:zoom], 'k-', linewidth=2, label='Real (Target)')
plt.plot(y_test.index[:zoom], y_pred_gb[:zoom], 'g--', linewidth=1.5, label='Gradient Boosting')
plt.plot(y_test.index[:zoom], y_pred_ridge[:zoom], 'r:', linewidth=1.5, label='Ridge Regression')
plt.title('Comparação: Real vs Previsto (Zoom 60 dias)')
plt.ylabel('Consumo (kWh)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico 2: Feature Importance (Interpretabilidade) ---
# Usando o Gradient Boosting para ver o que pesou mais
feature_importance = gb.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.title('Importância das Variáveis (Gradient Boosting)')
plt.xlabel('Importância Relativa')
plt.tight_layout()
plt.show()

# --- Gráfico 3: Análise de Resíduos vs Temperatura ---
# Para provar que o modelo falha mais em extremos (Secção 9 do relatório)
residuos = y_test - y_pred_gb
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['temp_media'], y=np.abs(residuos), alpha=0.6, hue=np.abs(residuos), palette='viridis')
plt.axhline(0, color='grey', linestyle='--')
plt.title('Erro Absoluto vs Temperatura Média')
plt.xlabel('Temperatura Média (°C)')
plt.ylabel('Erro Absoluto (kWh)')
plt.text(X_test['temp_media'].min(), np.abs(residuos).max(), "Nota: Erros tendem a aumentar\nnos extremos térmicos", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

print("\nConclusão: O Gradient Boosting capturou melhor as não-linearidades (picos de frio/calor) comparado ao Ridge.")
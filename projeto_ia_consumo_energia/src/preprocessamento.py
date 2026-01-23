import pandas as pd
import numpy as np

def gerar_dados_simulados(n_registos=15000):
    """
    Motor de Simulação de Redes Inteligentes.
    Gera dados sintéticos com sazonalidade, não-linearidades e ruído controlado.
    """
    np.random.seed(42)
    datas = pd.date_range(start='1983-01-01', periods=n_registos, freq='D')
    df = pd.DataFrame({'data': datas})
    t = np.arange(n_registos)

    # 1. Temperatura REAL (Física - o que realmente aconteceu)
    # Sazonalidade anual (senoide)
    temp_sazonal = 10 * np.sin(2 * np.pi * t / 365)
    # Temperatura base + ruído diário natural
    df['temp_real'] = 15 + temp_sazonal + np.random.normal(0, 3, n_registos)

    # 2. Temperatura PREVISTA (Informação disponível para os modelos "Reais")
    # Adicionamos ruído extra para simular o erro da previsão meteorológica (forecast error)
    erro_previsao_meteo = np.random.normal(0, 2.5, n_registos) 
    df['temp_previsao'] = df['temp_real'] + erro_previsao_meteo

    # 3. Consumo (Depende da Temp REAL, pois as pessoas reagem ao frio/calor que sentem)
    base = 200
    # Efeito em U: Gasta-se mais no frio (<18) e no calor (>18)
    efeito_temp = 1.5 * (df['temp_real'] - 18)**2 
    
    # Efeito Fim de Semana
    dia_semana = df['data'].dt.dayofweek
    efeito_fds = np.where(dia_semana >= 5, -50, 20)
    
    # Tendência de longo prazo (eletrificação)
    tendencia = t * 0.005
    
    # Consumo Final = Soma componentes + Ruído aleatório do sistema
    df['consumo'] = base + efeito_temp + efeito_fds + tendencia + np.random.normal(0, 10, n_registos)
    
    # Injeção de Outliers (Eventos raros: 0.5% dos dias)
    idx_outliers = np.random.choice(n_registos, size=int(n_registos*0.005), replace=False)
    df.loc[idx_outliers, 'consumo'] += np.random.uniform(100, 200, len(idx_outliers))
    
    df.set_index('data', inplace=True)
    return df

def criar_features(df_input):
    """
    Gera as features temporais e lags para os modelos.
    """
    df = df_input.copy()
    
    # Variáveis Temporais
    df['mes'] = df.index.month
    df['dia_semana'] = df.index.dayofweek
    df['fim_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    # Lags (Autocorrelação)
    for lag in [1, 3, 7]:
        df[f'lag_{lag}'] = df['consumo'].shift(lag)
        
    # Médias Móveis (Rolling Window)
    # IMPORTANTE: Usar shift(1) antes do rolling para evitar Data Leakage (olhar para o futuro)
    df['rolling_mean_7'] = df['consumo'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['consumo'].shift(1).rolling(window=7).std()
    
    # Remove os primeiros dias que ficaram com NaN devido aos lags/rolling
    return df.dropna()
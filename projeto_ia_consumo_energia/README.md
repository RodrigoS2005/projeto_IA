Previsão de Consumo Energético com Séries Temporais e Machine Learning
Resumo

Este repositório apresenta um estudo comparativo de modelos estatísticos e de Machine Learning aplicados à previsão de consumo elétrico diário, utilizando dados sintéticos que reproduzem padrões observados em sistemas energéticos reais.

O trabalho cobre todo o pipeline analítico, desde a geração de dados, passando pela engenharia de variáveis, treino de modelos, avaliação quantitativa, até à análise interpretativa dos resultados.

Objetivos

Simular dados energéticos realistas com sazonalidade, tendência e eventos extremos

Aplicar técnicas adequadas de engenharia de features para séries temporais

Comparar modelos baseline, modelos de ML e modelos estatísticos clássicos

Avaliar o desempenho preditivo usando métricas padrão

Analisar interpretabilidade e comportamento dos erros

Estrutura do Repositório
├── main.py                # Script principal
├── README.md              # Documentação do projeto
└── outputs/               # (opcional) Gráficos e resultados

Geração de Dados

Os dados são totalmente sintéticos e gerados programaticamente, com as seguintes características:

Frequência diária (15.000 observações)

Sazonalidade anual e semanal

Relação não-linear entre temperatura e consumo (forma em U)

Tendência de longo prazo

Outliers representando eventos extremos

Variáveis Geradas

Temperatura média, mínima e máxima

Humidade relativa

Consumo elétrico (variável alvo)

Engenharia de Features

Foram criadas variáveis explicativas adequadas a séries temporais:

Variáveis Temporais

Mês

Dia da semana

Indicador de fim de semana

Autocorrelação (Lags)

Consumo em t−1, t−3 e t−7

Estatísticas Móveis

Média móvel (7 dias)

Desvio padrão móvel (7 dias)

⚠️ Todas as estatísticas móveis utilizam shift(1) para evitar data leakage.

Metodologia Experimental
Separação Treino/Teste

Separação temporal sem baralhamento

Últimos 365 dias reservados para teste

Simulação de cenário real de previsão futura

Normalização

StandardScaler aplicado apenas aos dados de treino

Mesma escala aplicada ao conjunto de teste

Modelos Avaliados
Baselines

Média histórica

Persistência (Naive)

Machine Learning

Ridge Regression

Random Forest Regressor

Gradient Boosting Regressor

Modelo Estatístico

SARIMAX com variáveis exógenas

Ordem (1,1,1), sazonalidade semanal (0,1,1,7)

Treino limitado aos últimos ~700 dias por eficiência computacional

Métricas de Avaliação

MAE — Mean Absolute Error

RMSE — Root Mean Squared Error

MAPE — Mean Absolute Percentage Error

Os resultados são apresentados em formato tabular para comparação direta entre modelos.

Visualizações

O script gera automaticamente:

Comparação temporal entre valores reais e previstos

Importância das variáveis (Gradient Boosting)

Análise do erro absoluto em função da temperatura

Estas visualizações permitem avaliar desempenho, interpretabilidade e limitações dos modelos.

Resultados Principais

Modelos baseline apresentam desempenho limitado

O Ridge Regression melhora com features temporais, mas é limitado em não-linearidades

Modelos ensemble apresentam melhor desempenho global

O Gradient Boosting obtém os menores erros médios

Os erros aumentam em condições térmicas extremas

Dependências
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

Execução
python main.py

Limitações

Dados sintéticos (não representam um sistema real específico)

Avaliação baseada num único holdout temporal

SARIMAX treinado num subconjunto dos dados

Trabalhos Futuros

Validação cruzada temporal (rolling forecast origin)

Previsão multi-horizonte

Modelos baseados em boosting avançado (XGBoost, LightGBM)

Métodos de interpretabilidade como SHAP

Aplicação a dados reais de consumo energético

Autor

Projeto desenvolvido para fins académicos e educacionais, no contexto de análise de séries temporais e ciência de dados aplicada à energia.

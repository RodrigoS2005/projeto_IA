Este projeto foi feito para a cadeira de Inteligência Artificial. A nossa ideia foi criar um sistema que consiga prever quanto é que se vai gastar de eletricidade num dia, dependendo do tempo (temperatura e humidade) e se é dia de semana ou fim de semana.

 O que usamos e como correr:
Instalámos as bibliotecas básicas que demos nas aulas: pandas, numpy, matplotlib, seaborn, scikit-learn e statsmodels.

Para verem o código a funcionar e gerar os gráficos, basta correr:
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
python main.py

O que fizemos (Metodologia):
Como não tínhamos um dataset real à mão com todas as variáveis que queríamos, criámos um simulador que gera dados realistas desde 1983. Tentámos que os dados tivessem "ruído" e outliers para ser mais difícil de prever.

Baselines: Usámos a média e o valor do dia anterior para ver o erro mínimo.

Oráculo: Fizemos um modelo que já sabe a temperatura real do dia seguinte para ver qual seria o limite máximo de precisão.

Modelos de ML: Testámos Ridge, Random Forest e Gradient Boosting. O Gradient Boosting foi o que se portou melhor porque consegue perceber que gastamos mais luz tanto quando está muito frio como quando está muito calor.

Análise dos Resultados:
A parte mais interessante foi ver onde o modelo falha. Reparámos que quando há temperaturas muito extremas, o erro sobe bastante. Isto faz sentido porque as pessoas mudam os hábitos de repente quando o tempo aperta.

O nosso grupo é composto por: 

Autores: João Reis; Rodrigo Simões; José Lourenço; Tiago Ramos; Tomás Fernandes 

Números: 30010484; 30012765; 30013434; 30012727; 30013307

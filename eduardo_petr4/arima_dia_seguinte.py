# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Leitura e pré-processamento dos dados
#    - Fonte: série histórica PETR4 (04/01/2021–18/03/2025) obtida de CSV
#    - Converte "Último" de string BR (pontos e vírgulas) para float
#    - Inverte a ordem para que índice 0 seja o dia mais antigo
df = pd.read_csv('PETR4 Dados Históricos (1).csv', delimiter=',', encoding='utf-8')
df['Preço'] = (
    df['Último']
      .str.replace('.', '', regex=False)   # remove separador de milhares
      .str.replace(',', '.', regex=False)  # converte vírgula→ponto
      .astype(float)
)
serie   = df['Preço'].values[::-1]
n       = len(serie)

# 2. Divisão treino/teste (Box–Jenkins)
#    - Treino: primeiros 70% → usado para identificar e ajustar o ARIMA
#    - Teste: últimos 30% → validação out-of-sample em horizonte de curto prazoa
n_train = int(0.7 * n)
train   = serie[:n_train]
test    = serie[n_train:]

# 3. Determinação de d via ADF
#    - Teste de Dickey–Fuller Aumentado confirma necessidade de diferenciação
#    - H0: raiz unitária (não estacionária). p-value > 0.05 → aplica d=1
adf_p = adfuller(train)[1]
d     = 0 if adf_p < 0.05 else 1

# 4. Seleção de p e q (ACF/PACF)
#    - p=1: captura dependência de 1 lag autorregressivo
#    - q=1: captura erro de média móvel de 1 dia anterior
#    - h=3: horizonte de previsão de 3 dias (curto prazo recomendado)
p, q = 1, 1
h    = 3

# 5. Rolling forecast “3-ahead” com retraining diário
#    - Em cada passo, reajusta ARIMA(1,d,1) apenas com dados passados
#    - Forecast de 3 etapas, mas compara apenas o 3º passo (h-1)
#    - Incorpora valor real do dia t antes de prever t+1…t+3
history     = list(train)
predictions = []

for t in range(len(test) - h + 1):
    model = ARIMA(history, order=(p, d, q), trend='n').fit()
    fc    = model.forecast(steps=h)
    yhat  = fc[h-1]
    predictions.append(yhat)
    history.append(test[t])

predictions = np.array(predictions)

# 6. Cálculo de métricas de erro (MAE e RMSE)
#    - real_vals alinha test[h-1:] para comparação direta
real_vals = test[h-1:]
mae  = mean_absolute_error(real_vals, predictions)
rmse = np.sqrt(mean_squared_error(real_vals, predictions))
print(f"MAE (3-dias-à-frente rolling):  {mae:.4f}")
print(f"RMSE (3-dias-à-frente rolling): {rmse:.4f}")

# 7. Preparação de eixos para plot alinhado
#    - all_x: eixo completo da série (para contexto)
#    - hist_x: últimos 10 dias antes do teste, para comparação com TCN (Lea et al., 2016)
#    - test_x e fc_x: alinham previsões e valores reais no mesmo dia da série
all_x  = np.arange(n)
window = 10
hist_x = np.arange(n_train - window, n_train)
test_x = np.arange(n_train, n_train + len(test))
fc_x   = test_x[h-1:]  # desloca h-1 para alinhar 3-ahead ao dia certo

# 8. Plotagem (visualização em múltiplas escalas)
#    - Fundo cinza: série completa (semelhante ao “multi-scale” do TCN)
#    - Azul: zoom nos últimos 10 dias de treino (contexto local)
#    - Preto: valores reais para 3-ahead, no mesmo dia que o ponto previsto
#    - Vermelho: previsões de 3-ahead, mostrando acurácia de curto prazo
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(all_x, serie, color='lightgray', label='Série completa')
ax.plot(hist_x, serie[n_train-window:n_train], color='blue',
        label=f'({window} dias da série original antes de começar o teste)')
ax.scatter(fc_x, real_vals, color='black', zorder=5,
           label='Real (3 dias à frente)')
ax.plot(fc_x, predictions, '--x', color='red',
        label='Forecast 3 dias à frente')
# Desenha a linha tracejada marcando o fim do treino e já define o label
ax.axvline(
    n_train - 0.5,
    color='gray',
    linestyle='--',
    label='Fim dos 70% de treinamento'
)
ax.set_xlim(n_train - window, fc_x[-1] + 1)

# Ajuste de eixo y com pequena margem
all_plot = np.concatenate([serie[n_train-window:n_train], real_vals, predictions])
ymin, ymax = all_plot.min(), all_plot.max()
marg = (ymax - ymin) * 0.05
ax.set_ylim(ymin - marg, ymax + marg)
ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Legendas e título resumem parâmetros e relacionam ao framework TCN
ax.set_xlabel('Dias (índice)')
ax.set_ylabel('Preço (R$)')
ax.set_title(
    f'ARIMA({p},{d},{q}) rolling · previsão 3 dias à frente\n'
    f'MAE: {mae:.4f} | RMSE: {rmse:.4f}\n'
)
ax.legend()
plt.tight_layout()
plt.savefig('rolling_3dias_alinhado.png')
plt.close()

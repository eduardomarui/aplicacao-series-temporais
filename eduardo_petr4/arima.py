# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Leitura e pré-processamento
# ------------------------------------------------------------------
df = pd.read_csv('PETR4 Dados Históricos (1).csv', delimiter=',', encoding='utf-8')
df['Preço'] = (
    df['Último']
      .str.replace('.', '', regex=False)
      .str.replace(',', '.', regex=False)
      .astype(float)
)
serie    = df['Preço'].values[::-1]      # ordem cronológica (mais antigo → índice 0)
n        = len(serie)
n_train  = int(0.7 * n)
train    = serie[:n_train]
test_obs = serie[n_train:]               # toda a parte de teste

# 2. Definir d via ADF
# ------------------------------------------------------------------
adf_p = adfuller(train)[1]
d     = 0 if adf_p < 0.05 else 1

# 3. Parâmetros p, q e horizonte
# ------------------------------------------------------------------
p, q = 1, 1
h    = 4   # queremos 4 previsões one-step-ahead

# 4. Walk-forward forecasting
# ------------------------------------------------------------------
history     = list(train)
predictions = []
lower_ci    = []
upper_ci    = []

for t in range(h):
    model  = ARIMA(history, order=(p, d, q), trend='t')
    res    = model.fit()
    fc_obj = res.get_forecast(steps=1)
    yhat   = fc_obj.predicted_mean[0]
    ci     = fc_obj.conf_int(alpha=0.05)  # array 1×2: [lower, upper]

    predictions.append(yhat)
    lower_ci.append(ci[0, 0])
    upper_ci.append(ci[0, 1])

    # incorpora o valor real para a próxima iteração
    history.append(test_obs[t])

predictions = np.array(predictions)
lower_ci    = np.array(lower_ci)
upper_ci    = np.array(upper_ci)

# 5. Cálculo das métricas de erro sobre os h passos
# ------------------------------------------------------------------
real_vals = test_obs[:h]
mae  = mean_absolute_error(real_vals, predictions)
rmse = np.sqrt(mean_squared_error(real_vals, predictions))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# 6. Preparar dados para plot
# ------------------------------------------------------------------
window = 10
start  = max(0, n_train - window)
all_x    = np.arange(n)
hist_x   = np.arange(start, n_train)
fc_x     = np.arange(n_train, n_train + h)
hist_y   = serie[start:n_train]
real_y   = real_vals

# 7. Plotagem com zoom e formatação em centavos
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10,5))

# fundo da série completa
ax.plot(all_x, serie, color='lightgray', label='Série completa')

# últimos window dias de histórico
ax.plot(hist_x, hist_y, color='blue', label=f'Histórico (últimos {window} dias)')

# valores reais dos h dias
ax.plot(fc_x, real_y, 'o-', color='black', label=f'Real ({h} dias)')

# previsões walk-forward
ax.plot(fc_x, predictions, '--o', color='red', label='Forecast walk-forward')

# intervalo de confiança 95%
ax.fill_between(fc_x, lower_ci, upper_ci, alpha=0.3, label='IC 95%')

# linha de corte e zoom horizontal
ax.axvline(n_train - 1, color='gray', linestyle=':')
ax.set_xlim(start, n_train + h)

# zoom no eixo Y e formatação com duas casas decimais
all_plot = np.concatenate([hist_y, real_y, predictions, lower_ci, upper_ci])
ymin, ymax = all_plot.min(), all_plot.max()
marg       = (ymax - ymin) * 0.05
ax.set_ylim(ymin - marg, ymax + marg)
ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax.set_xlabel('Dias (índice)')
ax.set_ylabel('Preço (R$)')
ax.set_title(f'ARIMA({p},{d},{q}) walk-forward — Últimos {window} dias + {h} dias forecast')
ax.legend()
plt.tight_layout()

plt.savefig('forecast_walkforward_zoom.png')
plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# Implementação de ARIMA(2, d, 1) para previsão multi‑passo na série PETR4
# =============================================================================

# 1. Leitura e pré‑processamento da série histórica
# -------------------------------------------------
# - 'PETR4 Dados Históricos (1).csv' contém datas e preços no formato brasileiro.
# - Convertemos a coluna 'Último' para float: removemos separadores de milhar ('.')
#   e trocamos vírgulas por pontos para decimais.
df = pd.read_csv('PETR4 Dados Históricos (1).csv', delimiter=',', encoding='utf-8')
df['Preço'] = (df['Último']
               .str.replace('.', '', regex=False)  # retira pontos de milhar
               .str.replace(',', '.', regex=False) # vírgula → ponto decimal
               .astype(float))

# Inverte para ordem cronológica (do mais antigo para o mais recente)
serie = df['Preço'].values[::-1]

# 2. Divisão treino/teste (70% / 30%)
# -----------------------------------
# - Treino: primeiras 70% observações → usado para ajustar (fit) o ARIMA.
# - Teste: últimas 30% observações → usado para avaliar o forecast multi‑passo.
n = len(serie)
n_train = int(0.7 * n)
train, test = serie[:n_train], serie[n_train:]

# 3. Teste de estacionariedade (ADF) para definir d
# -------------------------------------------------
# - ARIMA requer série estacionária: sem tendência ou raiz unitária.
# - O parâmetro d é a "ordem de integração": número de vezes que se aplica diferenciação
#   para estabilizar média e variância da série.
# - Usamos o teste de Dickey–Fuller Aumentado (ADF):
#     * H0: existe raiz unitária (série não estacionária).
#     * p-value < 0.05 → rejeita H0 → série estacionária → d = 0 (sem diferenciação).
#     * p-value ≥ 0.05 → não estacionária → escolhemos d = 1 (diferenciação simples).
adf_p = adfuller(train)[1]
d     = 0 if adf_p < 0.05 else 1  # d = 0 ou 1 conforme estacionariedade

# 4. Escolha manual de p e q com base em ACF/PACF (ou conhecimento prévio)
# -------------------------------------------------------------------------
# - p (AR order): número de lags autorregressivos → capturam dependência dos p últimos lags.
#   Definimos p = 2 após observar autocorrelação parcial significativa até lag 2.
# - q (MA order): número de lags do termo de média móvel → modela dependência de erros passados.
#   Definimos q = 1 para incluir apenas o erro de previsão do passo anterior.
p, q = 2, 1

# 5. Ajuste do modelo ARIMA
# --------------------------
# - ARIMA(p, d, q) com trend='t' adiciona um termo de "drift" (tendência linear) no nível.
# - O drift é um coeficiente que permite ao modelo incorporar uma inclinação geral,
#   projetando crescimento ou decréscimo contínuo, em vez de convergir sempre à média.
# - Os coeficientes AR (phi_1, phi_2), MA (theta_1) e drift (mu) são estimados
#   por máxima verossimilhança usando todo o conjunto de treino.
model = ARIMA(train, order=(p, d, q), trend='t')
res   = model.fit()

# 6. Forecast multi‑passo
# -----------------------
# - forecast(steps=h) projeta h = len(test) pontos de uma só vez,
#   sem re‑alimentar o modelo com observações reais de teste.
h        = len(test)
forecast = res.forecast(steps=h)

# 7. Avaliação de desempenho
# ---------------------------
# - MAE: erro médio absoluto (unidades da série, R$).
# - RMSE: raiz do erro quadrático médio, penaliza mais discrepâncias grandes.
mae  = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Multi‑passo ARIMA({p},{d},{q}) — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 8. Visualização e salvamento do gráfico
# ----------------------------------------
# - Compara série completa, treino, teste e previsões.
# - plt.savefig() gera arquivo sem exibir na tela.
plt.figure(figsize=(10, 5))
plt.plot(serie,                       label='Série Completa',            color='lightgray')
plt.plot(np.arange(n_train), train,   label='Treino (70%)',             color='blue')
plt.plot(np.arange(n_train, n), test, label='Teste (30%)',              color='black')
plt.plot(np.arange(n_train, n_train+h), forecast,
         label='Forecast Multi‑Passo', linestyle='--', color='red')
plt.axvline(n_train, color='gray', linestyle=':')
plt.legend()
plt.xlabel('Dias')
plt.ylabel('Preço (R$)')
plt.title(f'Forecast Multi‑Passo ARIMA({p},{d},{q})')
plt.savefig('forecast_multipasso.png')
plt.close()

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Definição da arquitetura TCN
# ------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output

# 2. Função para criar sequências de dados
# ------------------------------------------------------------------
def create_sequences(data, seq_length, forecast_horizon):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        seq = data[i:i + seq_length]
        target = data[i + seq_length + forecast_horizon - 1]  # 3 dias à frente
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 3. Leitura e pré-processamento dos dados (mesmo que ARIMA)
# ------------------------------------------------------------------
df = pd.read_csv('PETR4 Dados Históricos (1).csv', delimiter=',', encoding='utf-8')
df['Preço'] = (
    df['Último']
      .str.replace('.', '', regex=False)   # remove separador de milhares
      .str.replace(',', '.', regex=False)  # converte vírgula→ponto
      .astype(float)
)
serie = df['Preço'].values[::-1]
n = len(serie)

# 4. Divisão treino/teste (mesmo que ARIMA)
# ------------------------------------------------------------------
n_train = int(0.7 * n)
train = serie[:n_train]
test = serie[n_train:]

# 5. Normalização dos dados
# ------------------------------------------------------------------
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()
serie_scaled = scaler.transform(serie.reshape(-1, 1)).flatten()

# 6. Parâmetros da TCN (equivalentes ao ARIMA)
# ------------------------------------------------------------------
seq_length = 10  # janela de observação (equivalente ao AR)
h = 3           # horizonte de previsão de 3 dias (mesmo que ARIMA)
num_channels = [16, 32, 16]  # arquitetura da TCN (reduzida para ser mais rápida)
kernel_size = 3
dropout = 0.2
learning_rate = 0.01  # learning rate maior para convergir mais rápido
epochs = 50  # menos épocas para ser mais rápido

# 7. Rolling forecast com retraining (mesmo approach do ARIMA)
# ------------------------------------------------------------------
import time

history_scaled = list(train_scaled)
predictions_scaled = []
total_steps = len(test) - h + 1

print(f"Iniciando rolling forecast com TCN...")
print(f"Total de passos: {total_steps}")
print(f"Parâmetros: seq_length={seq_length}, epochs={epochs}, lr={learning_rate}")
print("=" * 60)

start_time = time.time()

for t in range(total_steps):
    step_start = time.time()
    print(f"Passo {t+1}/{total_steps} - ", end="", flush=True)
    
    # Criar sequências para treino
    current_train = np.array(history_scaled)
    X_train, y_train = create_sequences(current_train, seq_length, h)
    
    if len(X_train) < 10:  # precisa de pelo menos algumas amostras
        print("Poucas amostras, pulando...")
        continue
    
    print(f"Treino: {len(X_train)} amostras - ", end="", flush=True)
    
    # Preparar dados para PyTorch
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # [batch, seq, features]
    X_train_tensor = X_train_tensor.transpose(1, 2)  # [batch, features, seq] para Conv1d
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Criar e treinar modelo
    model = TCN(input_size=1, output_size=1, num_channels=num_channels, 
                kernel_size=kernel_size, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dataset e DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=min(32, len(X_train)), shuffle=True)
    
    # Treinamento
    model.train()
    train_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Mostrar progresso a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"E{epoch+1}", end="", flush=True)
    
    train_time = time.time() - train_start
    print(f" - Treino: {train_time:.1f}s - ", end="", flush=True)
    
    # Fazer previsão
    model.eval()
    with torch.no_grad():
        # Usar os últimos seq_length pontos para prever
        last_sequence = np.array(history_scaled[-seq_length:])
        X_pred = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)
        X_pred = X_pred.transpose(1, 2)
        prediction_scaled = model(X_pred).item()
        predictions_scaled.append(prediction_scaled)
    
    # Adicionar valor real ao histórico
    history_scaled.append(test_scaled[t])
    
    step_time = time.time() - step_start
    elapsed_total = time.time() - start_time
    avg_time_per_step = elapsed_total / (t + 1)
    eta = avg_time_per_step * (total_steps - t - 1)
    
    print(f"Passo: {step_time:.1f}s | ETA: {eta/60:.1f}min")

total_time = time.time() - start_time
print("=" * 60)
print(f"Tempo total: {total_time/60:.1f} minutos")

# 8. Desnormalizar previsões
# ------------------------------------------------------------------
predictions_scaled = np.array(predictions_scaled)
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

# 9. Cálculo de métricas de erro (mesmo que ARIMA)
# ------------------------------------------------------------------
real_vals = test[h-1:]
mae = mean_absolute_error(real_vals, predictions)
rmse = np.sqrt(mean_squared_error(real_vals, predictions))
print(f"MAE (3-dias-à-frente rolling TCN):  {mae:.4f}")
print(f"RMSE (3-dias-à-frente rolling TCN): {rmse:.4f}")

# 10. Preparação de eixos para plot (mesmo que ARIMA)
# ------------------------------------------------------------------
all_x = np.arange(n)
window = 10
hist_x = np.arange(n_train - window, n_train)
test_x = np.arange(n_train, n_train + len(test))
fc_x = test_x[h-1:]  # desloca h-1 para alinhar 3-ahead ao dia certo

# 11. Plotagem (mesmo estilo que ARIMA)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(all_x, serie, color='lightgray', label='Série completa')
ax.plot(hist_x, serie[n_train-window:n_train], color='blue',
        label=f'({window} dias da série original antes de começar o teste)')
ax.scatter(fc_x, real_vals, color='black', zorder=5,
           label='Real (3 dias à frente)')
ax.plot(fc_x, predictions, '--x', color='red',
        label='Forecast TCN 3 dias à frente')

# Linha tracejada marcando fim do treino
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

# Legendas e título
ax.set_xlabel('Dias (índice)')
ax.set_ylabel('Preço (R$)')
ax.set_title(
    f'TCN rolling · previsão 3 dias à frente\n'
    f'MAE: {mae:.4f} | RMSE: {rmse:.4f}\n'
    f'Seq: {seq_length} | Channels: {num_channels} | Kernel: {kernel_size}'
)
ax.legend()
plt.tight_layout()
plt.savefig('tcn_rolling_3dias_alinhado.png')
plt.close()

print("Gráfico salvo como 'tcn_rolling_3dias_alinhado.png'")

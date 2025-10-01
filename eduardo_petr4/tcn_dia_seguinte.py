
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
from datetime import datetime, timedelta, timedelta
import random

# Definir seeds para reprodutibilidade
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Definição da arquitetura TCN
# ------------------------------------------------------------------
# A TCN é uma alternativa moderna às RNNs/LSTMs que usa convoluções
# dilatadas para capturar dependências temporais de longo prazo
class TemporalBlock(nn.Module):
    """
    Bloco temporal básico da TCN.
    
    Implementa duas camadas convolucionais 1D com:
    - Dilatação para capturar dependências de longo prazo
    - Padding causal (não vê o futuro) 
    - Conexões residuais para evitar vanishing gradient
    - Dropout para regularização
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Primeira camada convolucional com dilatação
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Remove padding extra para manter causalidade
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Segunda camada convolucional (mesmo padrão)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequência completa das operações
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Conexão residual (adapta dimensões se necessário)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass com conexão residual.
        out = ReLU(F(x) + x) onde F(x) é a transformação convolucional
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # Conexão residual + ativação final

class Chomp1d(nn.Module):
    """
    Remove padding do final da sequência para manter causalidade.
    
    A TCN usa padding para manter o tamanho da sequência, mas precisa
    remover o padding do final para garantir que não "veja o futuro".
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Remove chomp_size elementos do final da dimensão temporal
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """
    Rede Convolucional Temporal completa.
    
    Empilha múltiplos TemporalBlocks com dilatação exponencial:
    - Camada 1: dilatação = 1 (vê 1 passo)
    - Camada 2: dilatação = 2 (vê 2 passos)  
    - Camada 3: dilatação = 4 (vê 4 passos)
    
    Isso permite capturar dependências de diferentes escalas temporais.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Dilatação exponencial: 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Adiciona um TemporalBlock com dilatação específica
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                   dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    """
    Modelo TCN completo para previsão de séries temporais.
    
    Combina a rede convolucional temporal com uma camada linear final
    para produzir a previsão escalar (preço futuro).
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # Rede convolucional temporal (extrai features temporais)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        # Camada linear final (converte features em previsão)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Passa pela TCN e pega apenas o último timestep
        output = self.tcn(x)
        # Aplica camada linear ao último timestep para obter previsão
        output = self.linear(output[:, :, -1])
        return output

# 2. Função para criar sequências de dados
# ------------------------------------------------------------------
def create_sequences(data, seq_length, forecast_horizon):
    """
    Cria sequências de entrada e saída para treinar a TCN.
    
    Parâmetros:
    - data: série temporal normalizada
    - seq_length: tamanho da janela de observação (10 dias)
    - forecast_horizon: horizonte de previsão (3 dias à frente)
    
    Retorna:
    - sequences: janelas de seq_length dias 
    - targets: valores 3 dias à frente de cada janela
    
    Exemplo: se seq_length=10 e forecast_horizon=3:
    - Entrada: dias [1,2,3,4,5,6,7,8,9,10]
    - Saída: dia 13 (10 + 3)
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        seq = data[i:i + seq_length]  # janela de observação
        target = data[i + seq_length + forecast_horizon - 1]  # 3 dias à frente
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 3. Leitura e pré-processamento dos dados (mesmo que ARIMA)
# ------------------------------------------------------------------
# Carrega dados históricos das empresas exportados do investing.com

# Lista de empresas a analisar
empresas = ['PETR4', 'RECV3', 'WEG3']

# Mapeamento de arquivos CSV para cada empresa
arquivos_csv = {
    'PETR4': 'PETR4 Dados Históricos (1).csv',
    'RECV3': 'RECV3 Dados Históricos - cópia.csv',
    'WEG3': 'WEGE3 Dados Históricos (3).csv'  # Corrigido
}

# Loop sobre cada empresa
for empresa in empresas:
    print(f"\n🏢 Analisando empresa: {empresa}")
    print("=" * 80)
    
    # Carregar dados da empresa atual
    df = pd.read_csv(arquivos_csv[empresa], delimiter=',', encoding='utf-8')
    
    # Converte coluna 'Último' (preço de fechamento) de formato brasileiro para float
    # Formato brasileiro: "12.345,67" → formato python: 12345.67
    df['Preço'] = (
        df['Último']
          .str.replace('.', '', regex=False)   # remove separador de milhares (.)
          .str.replace(',', '.', regex=False)  # converte vírgula decimal (,) para ponto (.)
          .astype(float)
    )
    
    # Inverte ordem dos dados: mais antigo (índice 0) → mais recente (índice -1)
    # Necessário porque dados do investing.com vêm do mais recente para o mais antigo
    serie = df['Preço'].values[::-1][-360:]  # Últimos 360 dias
    n = len(serie)
    
    # 4. Divisão treino/teste (mesmo que ARIMA)
    # ------------------------------------------------------------------
    # Usa mesma divisão 70/30 que o ARIMA para comparação justa
    n_train = int(0.7 * n)      # 70% para treino
    train = serie[:n_train]     # dados de treino
    test = serie[n_train:]      # dados de teste
    
    # 5. Normalização dos dados
    # ------------------------------------------------------------------
    # TCNs são sensíveis à escala dos dados, diferente do ARIMA
    # StandardScaler transforma os dados para média=0 e desvio=1
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()  # aprende média/desvio do treino
    test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()        # aplica mesma transformação no teste
    serie_scaled = scaler.transform(serie.reshape(-1, 1)).flatten()      # série completa normalizada
    
    # 6. Parâmetros da TCN (equivalentes ao ARIMA)
    # ------------------------------------------------------------------
    # Vamos testar diferentes tamanhos de sequência para comparar performance
    seq_lengths_to_test = [3,5,10,15]  # diferentes tamanhos de observação
    h = 3                       # horizonte de previsão de 3 dias (mesmo que ARIMA)
    num_channels = [16, 32, 16] # arquitetura da TCN: 3 camadas com 16→32→16 filtros
    kernel_size = 3             # tamanho do kernel convolucional (3 pontos temporais)
    dropout = 0.2              # taxa de dropout para regularização
    learning_rate = 0.01       # taxa de aprendizado (maior para convergir mais rápido)
    epochs = 30                # menos épocas para ser mais rápido nos testes
    
    # 7. Rolling forecast com retraining para diferentes tamanhos de sequência
    # ------------------------------------------------------------------
    # Testa diferentes tamanhos de observação: 15, 10, 5 e 3 dias
    import time
    
    print("Testando TCN com diferentes tamanhos de sequência...")
    print(f"Tamanhos a testar: {seq_lengths_to_test}")
    print(f"Horizonte de previsão: {h} dias")
    print("=" * 80)
    
    # Dicionário para armazenar todos os resultados para a tabela Excel
    all_results = {}
    
    # Loop sobre diferentes tamanhos de sequência
    for seq_length in seq_lengths_to_test:
        print(f"\n🔍 Testando com seq_length = {seq_length} dias")
        print("-" * 50)
        
        # Resetar histórico para cada teste
        history_scaled = list(train_scaled)
        predictions_scaled = []
        total_steps = len(test) - h + 1
        
        print(f"Total de passos: {total_steps}")
        print(f"Parâmetros: seq_length={seq_length}, epochs={epochs}, lr={learning_rate}")
        print("=" * 50)
        
        start_time = time.time()
        
        for t in range(total_steps):
            step_start = time.time()
            print(f"Passo {t+1}/{total_steps} - ", end="", flush=True)
            
            # Criar sequências para treino usando apenas dados históricos
            current_train = np.array(history_scaled)
            X_train, y_train = create_sequences(current_train, seq_length, h)
            
            # Verificação de segurança: precisa de amostras suficientes
            if len(X_train) < 5:  # mínimo de 5 amostras
                print("Poucas amostras, pulando...")
                continue
            
            print(f"Treino: {len(X_train)} amostras - ", end="", flush=True)
            
            # Preparar dados para PyTorch (conversão de formatos)
            X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # [batch, seq, features]
            X_train_tensor = X_train_tensor.transpose(1, 2)           # [batch, features, seq] para Conv1d
            y_train_tensor = torch.FloatTensor(y_train)
            
            # Criar e configurar novo modelo TCN para este passo
            model = TCN(input_size=1, output_size=1, num_channels=num_channels, 
                        kernel_size=kernel_size, dropout=dropout)
            criterion = nn.MSELoss()                                   # função de perda
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # otimizador
            
            # Dataset e DataLoader para treinamento em lotes
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            dataloader = DataLoader(dataset, batch_size=min(32, len(X_train)), shuffle=True)
            
            # Treinamento do modelo
            model.train()
            train_start = time.time()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()                    # zera gradientes
                    output = model(batch_X)                  # forward pass
                    loss = criterion(output, batch_y.unsqueeze(-1))  # output: [batch, 1], target: [batch, 1]
                    loss.backward()                          # backward pass
                    optimizer.step()                         # atualiza pesos
                    epoch_loss += loss.item()
                
                # Mostrar progresso a cada 10 épocas
                if (epoch + 1) % 10 == 0:
                    print(f"E{epoch+1}", end="", flush=True)
            
            train_time = time.time() - train_start
            print(f" - Treino: {train_time:.1f}s - ", end="", flush=True)
            
            # Fazer previsão 3 dias à frente
            model.eval()
            with torch.no_grad():
                # Usar os últimos seq_length pontos do histórico para prever
                last_sequence = np.array(history_scaled[-seq_length:])
                X_pred = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)
                X_pred = X_pred.transpose(1, 2)
                prediction_scaled = model(X_pred).item()  # obter previsão escalar
                predictions_scaled.append(prediction_scaled)
            
            # Incorporar valor real do dia atual ao histórico (mesmo que ARIMA)
            history_scaled.append(test_scaled[t])
            
            # Calcular estatísticas de tempo e ETA
            step_time = time.time() - step_start
            elapsed_total = time.time() - start_time
            avg_time_per_step = elapsed_total / (t + 1)
            eta = avg_time_per_step * (total_steps - t - 1)
            
            print(f"Passo: {step_time:.1f}s | ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start_time
        print("=" * 50)
        print(f"Tempo total para seq_length={seq_length}: {total_time/60:.1f} minutos")
        
        # 8. Desnormalizar previsões
        # ------------------------------------------------------------------
        predictions_scaled = np.array(predictions_scaled)
        predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        # 9. Cálculo de métricas de erro
        # ------------------------------------------------------------------
        real_vals = test[h-1:]  # valores reais alinhados (3 dias à frente)
        mae = mean_absolute_error(real_vals, predictions)
        rmse = np.sqrt(mean_squared_error(real_vals, predictions))
        print(f"MAE ({seq_length} dias → 3 à frente):  {mae:.4f}")
        print(f"RMSE ({seq_length} dias → 3 à frente): {rmse:.4f}")
        
        # Armazenar resultados para a tabela Excel
        all_results[seq_length] = {
            'predictions': predictions,
            'real_vals': real_vals,
            'mae': mae,
            'rmse': rmse,
            'total_time': total_time
        }
        
        # 10. Preparação de eixos para plot
        # ------------------------------------------------------------------
        all_x = np.arange(n)
        window = 10  # Mesmo que ARIMA
        hist_x = np.arange(n_train - window, n_train)
        test_x = np.arange(n_train, n_train + len(test))
        fc_x = test_x[h-1:]  # eixo das previsões (alinhado 3 dias à frente)
        
        # 11. GRÁFICO 1: APENAS BOLINHAS (estilo ARIMA)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10,5))
        
        # Últimos dias de treino para contexto local (azul)
        ax.plot(hist_x, serie[n_train-window:n_train], color='blue', label='Últimos 10 dias de treino')
        
        # Valores reais 3 dias à frente (linha com pontos pretos)
        ax.plot(fc_x, real_vals, 'o-', color='black', label='Teste (real)')
        
        # Previsões TCN (linha tracejada com pontos coloridos)
        colors = ['red', 'orange', 'green', 'purple']
        color_idx = seq_lengths_to_test.index(seq_length)
        ax.plot(fc_x, predictions, '--o', color=colors[color_idx], label=f'Previsão 3 dias à frente')
        
        # Linha vertical marcando fim do período de treino
        ax.axvline(n_train - 0.5, color='gray', linestyle='--', label='Fim do treino')
        
        # Zoom na região de interesse
        ax.set_xlim(n_train - window, fc_x[-1] + 1)
        
        # Ajuste automático do eixo Y com margem
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
            f'{empresa} - TCN Rolling Forecast: {seq_length} dias → 3 à frente\n'
            f'MAE: {mae:.4f} | RMSE: {rmse:.4f}'
        )
        ax.legend()
        
        plt.tight_layout()
        
        # Salvar primeiro gráfico (bolinhas)
        filename_dots = f'{empresa}_tcn_rolling_{seq_length}dias_bolinhas.png'
        plt.savefig(filename_dots, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfico salvo: '{filename_dots}'")
        print("=" * 80)
    
    print(f"\n🎉 Todos os testes para {empresa} concluídos!")
    print("Gráficos salvos:")
    for seq_length in seq_lengths_to_test:
        print(f"  📍 {empresa}_tcn_rolling_{seq_length}dias_bolinhas.png")
    
    # Gerar tabela Excel com médias a cada 3 dias
    print(f"\n📊 Gerando tabela Excel com médias a cada 3 dias para {empresa}...")
    
    # Criar uma data base hipotética (1 de março de 2025)
    data_base = datetime(2025, 3, 1)
    
    # Criar o arquivo Excel com múltiplas abas
    excel_filename = f'{empresa}_tcn_previsoes_medias_3dias.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        for seq_length in seq_lengths_to_test:
            print(f"  📋 Processando aba: {seq_length} dias...")
            
            # Obter dados deste seq_length
            predictions = all_results[seq_length]['predictions']
            real_vals = all_results[seq_length]['real_vals']
            mae = all_results[seq_length]['mae']
            rmse = all_results[seq_length]['rmse']
            
            # Calcular quantos grupos de 3 dias temos
            n_predictions = len(predictions)
            n_groups = n_predictions // 3  # grupos completos de 3
            
            # Listas para armazenar os resultados
            periodos = []
            medias_previsto = []
            medias_real = []
            diferencas_abs = []
            diferencas_perc = []
            
            for i in range(n_groups):
                # Índices para o grupo atual (3 dias)
                start_idx = i * 3
                end_idx = start_idx + 3
                
                # Datas do período
                data_inicio = data_base + timedelta(days=start_idx)
                data_fim = data_base + timedelta(days=end_idx - 1)
                periodo = f"{data_inicio.strftime('%d/%m/%y')}-{data_fim.strftime('%d/%m/%y')}"
                
                # Calcular médias do período
                media_prev = np.mean(predictions[start_idx:end_idx])
                media_real = np.mean(real_vals[start_idx:end_idx])
                
                # Calcular diferenças
                diff_abs = abs(media_prev - media_real)
                diff_perc = (diff_abs / media_real) * 100 if media_real != 0 else 0
                
                # Armazenar resultados
                periodos.append(periodo)
                medias_previsto.append(media_prev)
                medias_real.append(media_real)
                diferencas_abs.append(diff_abs)
                diferencas_perc.append(diff_perc)
            
            # Criar DataFrame para esta aba
            df_seq = pd.DataFrame({
                'Período': periodos,
                'Média Previsto (R$)': medias_previsto,
                'Média Real (R$)': medias_real,
                'Diferença Absoluta (R$)': diferencas_abs,
                'Diferença Percentual (%)': diferencas_perc
            })
            
            # Formatar valores monetários
            df_seq['Média Previsto (R$)'] = df_seq['Média Previsto (R$)'].round(2)
            df_seq['Média Real (R$)'] = df_seq['Média Real (R$)'].round(2)
            df_seq['Diferença Absoluta (R$)'] = df_seq['Diferença Absoluta (R$)'].round(2)
            df_seq['Diferença Percentual (%)'] = df_seq['Diferença Percentual (%)'].round(2)
            
            # Adicionar estatísticas resumo
            estatisticas = pd.DataFrame({
                'Métrica': ['MAE Geral', 'RMSE Geral', 'Média Diff. Abs.', 'Média Diff. %', 'Tempo (min)'],
                'Valor': [
                    f"{mae:.4f}",
                    f"{rmse:.4f}", 
                    f"{np.mean(diferencas_abs):.2f}",
                    f"{np.mean(diferencas_perc):.2f}%",
                    f"{all_results[seq_length]['total_time']/60:.1f}"
                ]
            })
            
            # Salvar na aba correspondente
            nome_aba = f"{seq_length}_dias"
            df_seq.to_excel(writer, sheet_name=nome_aba, index=False, startrow=0)
            
            # Adicionar estatísticas embaixo
            estatisticas.to_excel(writer, sheet_name=nome_aba, index=False, 
                                startrow=len(df_seq) + 3)
            
            print(f"    ✅ {len(df_seq)} períodos de 3 dias processados")
    
    print(f"\n📈 Tabela Excel salva como: '{excel_filename}'")
    print(f"📋 Abas criadas: {', '.join([f'{s}_dias' for s in seq_lengths_to_test])}")
    print("\n🎯 Estrutura da tabela:")
    print("  • Período: Data início-fim (dd/mm/yy)")
    print("  • Média Previsto: Média dos 3 dias previstos")
    print("  • Média Real: Média dos 3 dias reais")
    print("  • Diferença Absoluta: |Previsto - Real|")
    print("  • Diferença Percentual: (Diff. Abs. / Real) * 100")
    print("  • Estatísticas gerais no final de cada aba")
    print("=" * 100)

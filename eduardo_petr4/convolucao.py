import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para realizar average pooling com uma janela definida (pool_size)
def average_pooling(array, pool_size=2):
    # Para cada bloco de 'pool_size' elementos, calcula a média.
    n = len(array) - (len(array) % pool_size)  # Garante blocos completos
    pooled = [np.mean(array[i:i+pool_size]) for i in range(0, n, pool_size)]
    return np.array(pooled)

# Lê o CSV com os dados diários da PETR4
df = pd.read_csv('PETR4 Dados Históricos (1).csv', delimiter=',', encoding='utf-8')

# Exibe as primeiras linhas para verificação
print(df.head())

# Converte a coluna "Último" para valores numéricos:
# Remove os pontos (que servem como separadores de milhar) e substitui vírgulas por pontos para decimais.
df['Último'] = df['Último'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['Último'] = pd.to_numeric(df['Último'], errors='coerce')

# Extrai a série de preços diários
serie_original = df['Último'].values

# Os dados estão em ordem decrescente (do mais recente para o mais antigo)
# Inverte a série para ter a ordem cronológica (do passado para o presente)
serie_original = serie_original[::-1]

# ============================================================
# 1. Série Original
# ------------------------------------------------------------
# Análise:
# - Este gráfico mostra os preços diários da PETR4 sem nenhum processamento adicional.
# - Pode-se observar a volatilidade natural dos preços e o ruído diário.
# - Conclusão: É difícil identificar tendências de longo prazo devido às variações diárias.
# ============================================================

# Salva o gráfico da Série Original
plt.figure(figsize=(12, 6))
plt.plot(serie_original, label="Série Original", color='blue')
plt.title("PETR4 - Série Original (Ordem Cronológica)")
plt.xlabel("Observações (dias)")
plt.ylabel("Preço")
plt.legend()
plt.savefig("serie_original.png")
plt.close()

# ============================================================
# 2. Série Convolucionada (Média Móvel de 10 Períodos)
# ------------------------------------------------------------
# Aplicação da convolução:
# - Aqui usamos uma média móvel com janela de 10 períodos, que é uma forma simples de 
#   realizar uma convolução linear sobre os dados diários.
# - Esta operação suaviza a série, integrando os dados locais para destacar a tendência geral.
#
# Relação com o artigo:
# - No artigo, a convolução é realizada com filtros 1D que capturam variações temporais.
# - Nossa média móvel é uma versão simplificada: ao invés de filtros aprendidos com pesos,
#   usamos um filtro uniforme. Assim, não temos o parâmetro de aprendizado nem a não-linearidade,
#   mas o conceito de integrar a informação ao longo do tempo é o mesmo.
#
# Análise:
# - O gráfico convolucionado apresenta uma série suavizada, onde o ruído é reduzido.
# - Conclusão: É mais fácil identificar as tendências de longo prazo, mas detalhes locais foram atenuados.
# ============================================================

janela = 10
filtro = np.ones(janela) / janela
serie_convolucionada = np.convolve(serie_original, filtro, mode='same')

# Salva o gráfico da Série Convolucionada
plt.figure(figsize=(12, 6))
plt.plot(serie_convolucionada, label="Série Convolucionada (Média Móvel 10 períodos)", color='green')
plt.title("PETR4 - Série Convolucionada (Ordem Cronológica)")
plt.xlabel("Observações (dias)")
plt.ylabel("Preço")
plt.legend()
plt.savefig("serie_convolucionada.png")
plt.close()

# ============================================================
# 3. Série Convolucionada com Average Pooling
# ------------------------------------------------------------
# Aplicação do average pooling:
# - Após a convolução, aplicamos average pooling com uma janela de 2 períodos.
# - Essa operação agrupa os dados suavizados em blocos e calcula a média de cada bloco,
#   resultando em uma série com menor resolução temporal mas ainda mais suave.
#
# Análise:
# - O gráfico resultante apresenta uma visão ainda mais resumida da evolução dos preços.
# - Conclusão: O average pooling reforça a tendência de longo prazo, removendo detalhes menores,
#   facilitando a análise de comportamento geral dos preços.
# ============================================================

serie_convolucionada_pool = average_pooling(serie_convolucionada, pool_size=2)

# Salva o gráfico da Série Convolucionada com Average Pooling
plt.figure(figsize=(12, 6))
plt.plot(serie_convolucionada_pool, label="Convolucionada com Average Pooling (janela=2)", color='orange')
plt.title("PETR4 - Convolucionada com Average Pooling (Ordem Cronológica)")
plt.xlabel("Blocos de 2 períodos")
plt.ylabel("Preço")
plt.legend()
plt.savefig("serie_convolucionada_average_pool.png")
plt.close()

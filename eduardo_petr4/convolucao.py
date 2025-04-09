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
# Remove os pontos (separadores de milhar) e substitui vírgulas por pontos para decimais.
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
# - Usamos uma média móvel com janela de 10 períodos para suavizar a série.
# - Essa operação integra os dados locais para destacar a tendência geral.
#
# Relação com o artigo:
# - No artigo, a convolução com filtros 1D captura variações temporais.
# - Aqui, usamos um filtro uniforme (média móvel), que é uma versão simplificada
#   do conceito apresentado.
#
# Análise:
# - O gráfico suaviza os ruídos diários, facilitando a identificação de tendências de longo prazo.
# ============================================================

janela = 10
filtro = np.ones(janela) / janela
serie_convolucionada = np.convolve(serie_original, filtro, mode='same')

plt.figure(figsize=(12, 6))
plt.plot(serie_convolucionada, label="Série Convolucionada (Média Móvel 10 períodos)", color='green')
plt.title("PETR4 - Série Convolucionada (Ordem Cronológica)")
plt.xlabel("Observações (dias)")
plt.ylabel("Preço")
plt.legend()
plt.savefig("serie_convolucionada.png")
plt.close()

# ============================================================
# 3. Série Convolucionada com Average Pooling (1x)
# ------------------------------------------------------------
# Aplicação do average pooling:
# - Após a convolução, aplicamos average pooling com janela de 2 períodos.
# - Essa operação agrupa os dados e calcula a média de cada bloco, resultando em uma
#   resolução temporal menor, mas com maior suavização.
#
# Análise:
# - O gráfico apresenta uma visão resumida da evolução dos preços,
#   destacando as tendências gerais e removendo detalhes menores.
# ============================================================

serie_convolucionada_pool = average_pooling(serie_convolucionada, pool_size=2)

plt.figure(figsize=(12, 6))
plt.plot(serie_convolucionada_pool, label="Convolucionada com Average Pooling (1x)", color='orange')
plt.title("PETR4 - Convolucionada com Average Pooling (Ordem Cronológica)")
plt.xlabel("Blocos de 2 períodos")
plt.ylabel("Preço")
plt.legend()
plt.savefig("serie_convolucionada_average_pool.png")
plt.close()

# ============================================================
# 4. Série Convolucionada com 2x Average Pooling
# ------------------------------------------------------------
# Aplicação do average pooling duas vezes:
# - Aqui aplicamos o average pooling novamente à série já processada (1x),
#   reduzindo ainda mais a resolução temporal.
#
# Análise:
# - O gráfico apresenta uma versão extremamente suavizada da série,
#   enfatizando tendências muito de longo prazo, mas com perda de detalhes.
# ============================================================

serie_convolucionada_pool_2 = average_pooling(serie_convolucionada_pool, pool_size=2)

plt.figure(figsize=(12, 6))
plt.plot(serie_convolucionada_pool_2, label="Convolucionada com 2x Average Pooling", color='red')
plt.title("PETR4 - Convolucionada com 2x Average Pooling (Ordem Cronológica)")
plt.xlabel("Blocos (resolução reduzida)")
plt.ylabel("Preço")
plt.legend()
plt.savefig("serie_convolucionada_2x_average_pool.png")
plt.close()

# ============================================================
# 5. UpSampling (Interpolação) da Série 2x Average Pooling para a Resolução do 1x Average Pooling
# ------------------------------------------------------------
# Aplicação da interpolação:
# - Queremos "voltar" a resolução da série 2x average pooling para que ela tenha
#   o mesmo comprimento que a série obtida com 1x pooling.
# - Usamos interpolação linear (np.interp) para criar uma nova série, mantendo os
#   valores já existentes e preenchendo os espaços com valores interpolados.
#
# Análise:
# - O gráfico resultante compara a série de 1x pooling com a versão interpolada
#   a partir da 2x pooling.
# - Conclusão: Essa abordagem permite recuperar parte da resolução perdida,
#   combinando a suavidade do 2x pooling com a resolução do 1x pooling.
# ============================================================

# Índices do array de 2x pooling (menor resolução)
x_2x = np.arange(len(serie_convolucionada_pool_2))
# Novos índices para a resolução do 1x pooling
x_target = np.linspace(0, len(serie_convolucionada_pool_2) - 1, num=len(serie_convolucionada_pool))
# Realiza a interpolação linear
serie_interpolada = np.interp(x_target, x_2x, serie_convolucionada_pool_2)

plt.figure(figsize=(12, 6))
plt.plot(serie_convolucionada_pool, label="Convolucionada com Average Pooling (1x)", color='orange')
plt.plot(serie_interpolada, label="Interpolação do 2x Average Pooling", color='purple', linestyle='--')
plt.title("Comparação: 1x Average Pooling vs. Interpolação do 2x Average Pooling")
plt.xlabel("Blocos de 2 períodos (resolução do 1x pooling)")
plt.ylabel("Preço")
plt.legend()
plt.savefig("comparacao_interpolacao.png")
plt.close()

# ============================================================
# 6. UpSampling do UpSampling para Voltar à Resolução da Série Original
# ------------------------------------------------------------
# Aplicação da interpolação:
# - Agora, queremos "voltar" do upsampling (que trouxe a resolução para a do 1x pooling)
#   para a resolução original da série (com todos os pontos diários).
# - Para isso, aplicamos interpolação linear novamente, desta vez para mapear os dados
#   da resolução do 1x pooling para a resolução da série original.
#
# Análise:
# - O gráfico final compara três séries:
#    1. A Série Original (dados diários)
#    2. A Série após 1x Average Pooling (resolução reduzida, mas suavizada)
#    3. A Série interpolada a partir da 2x Average Pooling, upsampleada para a resolução da 1x pooling,
#       e depois re-interpolada para a resolução original.
# - Conclusão: Essa abordagem mostra como é possível recuperar parte da resolução perdida
#   após pooling, combinando a suavização com a interpolação para aproximar a série original.
# ============================================================

# 'serie_interpolada' já possui a resolução do 1x pooling (obtida por interpolar o 2x pooling).
# Vamos agora interpolar 'serie_interpolada' para ter a mesma resolução da 'serie_original'.
x_interp = np.arange(len(serie_interpolada))  # Índices da série interpolada (1x pooling)
x_target_orig = np.linspace(0, len(serie_interpolada) - 1, num=len(serie_original))
serie_interpolada_original = np.interp(x_target_orig, x_interp, serie_interpolada)

# Gráfico comparando as 3 séries:
plt.figure(figsize=(12, 6))
plt.plot(serie_original, label="Série Original (Diária)", color='blue')
plt.plot(serie_interpolada_original, label="Interpolação do 2x Average Pooling para Resolução Original", color='purple', linestyle='--')
plt.title("Comparação: Série Original vs. 1x Pooling vs. Interpolação para Resolução Original")
plt.xlabel("Observações (dias)")
plt.ylabel("Preço")
plt.legend()
plt.savefig("comparacao_final.png")
plt.close()
# ============================================================
# 7. Gráficos das Diferenças
# ------------------------------------------------------------
# 7.1 Diferença entre a interpolação do 2x pooling e a série 1x pooling
# ------------------------------------------------------------
# Dif = serie_interpolada - serie_convolucionada_pool
diff_1x = serie_interpolada - serie_convolucionada_pool

plt.figure(figsize=(12, 6))
plt.plot(diff_1x, label="Diferença: Interpolação 2x Pooling - Série 1x Pooling", color='brown')
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Diferença entre Interpolação do 2x Pooling e Série 1x Pooling")
plt.xlabel("Blocos de 2 períodos (resolução do 1x pooling)")
plt.ylabel("Diferença de Preço")
plt.legend()
plt.savefig("diferenca_interpolacao_1x.png")
plt.close()

# 7.2 Diferença entre a série re-interpolada para resolução original e a série original
# ------------------------------------------------------------
# Dif2 = serie_interpolada_original - serie_original
diff_orig = serie_interpolada_original - serie_original

plt.figure(figsize=(12, 6))
plt.plot(diff_orig, label="Diferença: Interpolação p/ Original - Série Original", color='magenta')
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Diferença entre Interpolação para Resolução Original e Série Original")
plt.xlabel("Observações (dias)")
plt.ylabel("Diferença de Preço")
plt.legend()
plt.savefig("diferenca_interpolacao_original.png")
plt.close()

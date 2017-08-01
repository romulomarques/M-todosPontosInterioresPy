import numpy as np

# Formato 1 de instancias para Metodos de Pontos Interiores:
# m
# (linha vazia)
# n
# (linha vazia)
# c_transposto
# (linha vazia)
# b_transposto
# (linha vazia)
# A

def lerInstanciaMPIFormato1(nomeArquivo):
    with open(nomeArquivo, 'r') as arquivoLeitura:
        numLinhas = int(arquivoLeitura.readline())

        # Lendo linha vazia que separa objetos
        arquivoLeitura.readline()

        numColunas = int(arquivoLeitura.readline())

        # Lendo linha vazia que separa objetos
        arquivoLeitura.readline()

        # Lendo vetor 'c' como uma grande string.
        vetorCzinho = arquivoLeitura.readline()
        # Dividindo a string 'c' numa lista de strings. Usamo o 'espaço' como separador.
        vetorCzinho = vetorCzinho.split()
        # Convertendo a lista de strings para uma lista de floats.
        vetorCzinho = [float(elemento) for elemento in vetorCzinho]
        # Transformando lista de floats num array da biblioteca numpy. Portanto, agora temos o vetor
        # 'c' na forma de array.
        vetorCzinho = np.array(vetorCzinho)
        # Representando o vetor 'c' na forma (n, 1).
        vetorCzinho = np.reshape(vetorCzinho, (np.shape(vetorCzinho)[0], 1))

        # Lendo linha vazia que separa objetos
        arquivoLeitura.readline()

        # Lendo vetor 'b' como uma grande string.
        vetorBzinho = arquivoLeitura.readline()
        # Dividindo a string 'b' numa lista de strings. Usamo o 'espaço' como separador.
        vetorBzinho = vetorBzinho.split()
        # Convertendo a lista de strings para uma lista de floats.
        vetorBzinho = [float(elemento) for elemento in vetorBzinho]
        # Transformando lista de floats num array da biblioteca numpy. Portanto, agora temos o vetor
        # 'b' na forma de array.
        vetorBzinho = np.array(vetorBzinho)
        # Representando o vetor 'b' na forma (m, 1).
        vetorBzinho = np.reshape(vetorBzinho, (np.shape(vetorBzinho)[0], 1))

        # Lendo linha vazia que separa objetos
        arquivoLeitura.readline()

        matrizA = arquivoLeitura.readlines()
        indiceLinhaA = 0
        for linhaA in matrizA:
            linhaA = linhaA.split()
            linhaA = [float(elemento) for elemento in linhaA]

            matrizA[indiceLinhaA] = linhaA
            indiceLinhaA = indiceLinhaA + 1

        matrizA = np.array(matrizA)

    arquivoLeitura.closed

    return numLinhas, numColunas, vetorCzinho, vetorBzinho, matrizA

if(__name__ == "__main__"):
    nomeArquivo = "Instancias\\instancia1.txt"
    m, n, c, b, A = lerInstanciaMPIFormato1(nomeArquivo)

    print("m")
    print(m)

    print("n")
    print(n)

    print("c")
    print(c)

    print("b")
    print(b)

    print("A")
    print(A)

#nomeArquivo = "Instancias\\vtp_base"

# Lendo todas as linhas do arquivo.
#with open(nomeArquivo, 'r') as arquivoLeitura:
    # Lendo a primeira linha do arquivo.
#    linhaArquivo = arquivoLeitura.readline()

    # Na primeira linha do arquivo, capturando:
    #   -> numero de linhas;
    #   -> numero de colunas;
    #   -> numero de elementos nao-zero na matriz A;
    #   -> valor a ser adicionando ao custo para conseguir o custo original do problema,
    # respectivamente.
#    elementosPrimeiraLinha = linhaArquivo.split()
#    numLinhas = int(elementosPrimeiraLinha[0])
#    numColunas = int(elementosPrimeiraLinha[1])
#    numNaoZeros = int(elementosPrimeiraLinha[2])
#    valorAddCusto = float(elementosPrimeiraLinha[3])

#    print("numero de linhas")
#    print(numLinhas)
#    print("numero de colunas")
#    print(numColunas)
#    print("numero de elementos nao nulos")
#    print(numNaoZeros)
#    print("valor para somar no custo para obtencao do custo original")
#    print(valorAddCusto)

    # Para cada coluna, vamos capturar o ponteiro do inicio dos indices das linhas que possuem elementos nao nulos.
#    indicesNaoZerosNasColunas = np.zeros(numColunas+1, int)
#    totalPonteiros = numColunas + 1
#    ponteiro = 0
#    while(ponteiro < totalPonteiros):
#        linhaArquivo = arquivoLeitura.readline()
#        indicesNaoZerosNasColunas[ponteiro] = int(linhaArquivo)
#        ponteiro = ponteiro + 1

#    print("indices das linhas dos elementos nao nulos nas colunas")
#    print(indicesNaoZerosNasColunas)

    # Para cada elemento nao nulo da matriz, vamos capturar o indice da sua linha.
#    indicesLinhas = np.zeros(numNaoZeros, int)
#    posicao = 0
#    while (posicao < numNaoZeros):
#        linhaArquivo = arquivoLeitura.readline()
#        indicesLinhas[posicao] = int(linhaArquivo)
#        posicao = posicao + 1

#    print("indice das linhas dos elementos nao nulos")
#    print(indicesLinhas)

    # Capturando os elementos nao nulos da matriz.
#    elementosNaoNulos = np.zeros(numNaoZeros)
#    posicao = 0
#    while (posicao < numNaoZeros):
#        linhaArquivo = arquivoLeitura.readline()
#        elementosNaoNulos[posicao] = float(linhaArquivo)
#        posicao = posicao + 1

#    print("elementos nao nulos")
#    print(elementosNaoNulos)

    # Capturando lado direito 'b'.
#    vetorB = np.zeros(numLinhas)
#    posicao = 0
#    while (posicao < numLinhas):
#        linhaArquivo = arquivoLeitura.readline()
#        vetorB[posicao] = float(linhaArquivo)
#        posicao = posicao + 1

#    print("b")
#    print(vetorB)

    # Capturando vetor de custos 'c'.
#    vetorC = np.zeros(numColunas)
#    posicao = 0
#    while (posicao < numColunas):
#        linhaArquivo = arquivoLeitura.readline()
#        vetorC[posicao] = float(linhaArquivo)
#        posicao = posicao + 1

#    print("c")
#    print(vetorC)

    # Capturando lower bounds.
#    lb = np.zeros(numColunas)
#   posicao = 0
#   while (posicao < numColunas):
#       linhaArquivo = arquivoLeitura.readline()
#       lb[posicao] = float(linhaArquivo)
#       posicao = posicao + 1

#    print("lower bounds")
#   print(lb)

    # Capturando upper bounds.
#    ub = np.zeros(numColunas)
#   posicao = 0
#   while (posicao < numColunas):
#       linhaArquivo = arquivoLeitura.readline()
#       ub[posicao] = float(linhaArquivo)
#       posicao = posicao + 1

#   print("upper bounds")
#   print(ub)

    #Acheia = np.zeros(numLinhas, numColunas)
    #while

#arquivoLeitura.closed
import numpy as np

def reordenarMatriz(matriz, vetorIndices):
    matrizReordenada = np.copy(matriz)
    cont = 0
    for linha in vetorIndices:
        matrizReordenada[cont] = matriz[linha]
        cont = cont + 1

    return matrizReordenada

def unitarizarDiag(matrizA, linhasIndices):
    cont = 0
    for elemento in linhasIndices:
        pivotCorrente = matrizA[elemento, cont]
        if (pivotCorrente != 0):
            matrizA[elemento, :] = matrizA[elemento, :] / pivotCorrente
        cont = cont + 1
    return matrizA

def unitarizarDiagGaussMatrizQuadrada(matrizA, linhasIndices):
    dimA = np.shape(matrizA)

    if(dimA[0] + 1 == dimA[1]):
        coluna = 0
        for linha in linhasIndices:
            pivot = matrizA[linha,coluna]
            if(pivot != 0):
                matrizA[linha,-1] = matrizA[linha,-1] / pivot
            coluna = coluna + 1

    return matrizA

# Usamos a linha de posicao 'linhaCorrente' para zerar a coluna de indice 'pivot'. O 'zeramento' acontece combinando a
# linha de posicao 'linhaCorrente' com cada uma outras linhas. Neste processo, no entanto, atualizamos somente algumas
# colunas de 'matrizA', ahquelas que estao descritas em 'colunasIndices'.
def zerarColunaGaussAlgumasColunas(matrizA, linhasIndices, colunasIndices, pivotCorrente):
    epsZero = 0

    # Quantidade de linhas analisadas.
    numLinhas = (np.shape(linhasIndices))[0]

    # Zerando as colunas 'linhaCorrente' de todas as linhas nas posicoes ABAIXO da linha de posicao 'linhaCorrente'. Note que, garantidamente,
    # o elemento pivot A[vetorIndices[linhaCorrente], linhaCorrente] eh nao nulo.
    linhaK = pivotCorrente + 1
    while (linhaK < numLinhas):
        if(abs(matrizA[linhasIndices[linhaK], pivotCorrente]) > epsZero):
            zerador = - (matrizA[linhasIndices[linhaK], pivotCorrente] / matrizA[linhasIndices[pivotCorrente], pivotCorrente])
            matrizA[linhasIndices[linhaK], pivotCorrente] = 0
            for elemento in colunasIndices:
                if(elemento > pivotCorrente):
                    matrizA[linhasIndices[linhaK], elemento] = matrizA[linhasIndices[linhaK], elemento] + zerador * matrizA[linhasIndices[pivotCorrente], elemento]

        linhaK = linhaK + 1

    # Zerando as colunas 'linhaCorrente' de todas as linhas nas posicoes ACIMA da linha de posicao 'linhaCorrente'. Note que, garantidamente,
    # o elemento pivot A[vetorIndices[linhaCorrente], linhaCorrente] eh nao nulo.
    linhaK = pivotCorrente - 1
    while (linhaK > -1):
        if(abs(matrizA[linhasIndices[linhaK], pivotCorrente]) > epsZero):
            zerador = - (matrizA[linhasIndices[linhaK], pivotCorrente] / matrizA[linhasIndices[pivotCorrente], pivotCorrente])
            matrizA[linhasIndices[linhaK], pivotCorrente] = 0
            for elemento in colunasIndices:
                if (elemento > pivotCorrente):
                    matrizA[linhasIndices[linhaK], elemento] = matrizA[linhasIndices[linhaK], elemento] + zerador * matrizA[linhasIndices[pivotCorrente], elemento]

        linhaK = linhaK - 1

def eliminacaoGaussSubMatriz(A, linhasIndices, colunasIndices, pivotInicial, numPivots, epsZero):
    # Dimensoes de A
    dimA = np.shape(A)

    # Total de linhas da matriz
    numLinhas = np.shape(linhasIndices)[0]

    # pivotInicial serah o primeiro pivot a ser analisado nessa aplicacao do metodo de Gauss.
    pivotCorrente = pivotInicial

    # numero de linhas que jah tiveram suas colunas 'pivotCorrente' zeradas.
    numPivotsAnalisados = 0

    # Booleano que nos diz se a matriz 'A' eh LD ou nao.
    ehLD = False

    # Aplicamos o algoritmo de Gauss nas 'numPivots' primeiras linhas de 'A'. Processo tambem conhecido por triangularizacao.
    while ( (numPivotsAnalisados < numPivots) and (not(ehLD)) ):
        linhaK = -1

        indiceLinhaMax = pivotCorrente
        valorAbsMax = abs(A[linhasIndices[pivotCorrente], pivotCorrente])

        # Primeira linha abaixo da linha corrente.
        linhaK = pivotCorrente + 1

        # Na coluna 'pivotCorrente', procuramos o elemento de maior valor absoluto abaixo do nosso pivot atual
        # para entao se tornar o novo pivot.
        while (linhaK < numLinhas):

            if (abs(A[linhasIndices[linhaK], pivotCorrente]) > valorAbsMax):
                indiceLinhaMax = linhaK
                valorAbsMax = abs(A[linhasIndices[linhaK], pivotCorrente])

            linhaK = linhaK + 1

        # Caso o elemento de maior valor absoluto seja pequeno o suficiente para ser considerado zero, dizemos que as
        # linhas da nossa matriz 'A' sao linearmente independentes.
        # Caso contrario, trocamos a linha do elemento de maior valor absoluto com a linha atual e entao zeramos toda a
        # coluna acima e abaixo do nosso pivot atual.
        if(valorAbsMax <= epsZero):
            ehLD = True
        else:
            linhaTroca = linhasIndices[pivotCorrente]
            linhasIndices[pivotCorrente] = linhasIndices[indiceLinhaMax]
            linhasIndices[indiceLinhaMax] = linhaTroca

            # zerando a coluna de indice 'pivotCorrente' em todas as linhas diferentes da linha de posicao 'linhaCorrente'.
            zerarColunaGaussAlgumasColunas(A, linhasIndices, colunasIndices, pivotCorrente)

        # Uma vez que zeramos a coluna de indice 'pivotCorrente' usando a linha na posicao 'pivotCorrente', vamos para a
        # proxima linha, a linha de posicao 'pivotCorrente + 1' que terah o elemento ['pivotCorrente + 1', 'pivotCorrente + 1']
        # como novo pivot.
        pivotCorrente = pivotCorrente + 1

        # Terminamos de analisar o pivot atual. Portanto, incrementamos o numero de pivots analisados.
        numPivotsAnalisados = numPivotsAnalisados + 1

    return (A, linhasIndices, ehLD)

if(__name__ == "__main__"):
    #A = np.array([[0.0, -1.0, 1.0, 2.0], [-1.0, 3.0, 0.0, 5.0], [2.0, 0.0, 6.0, 20.0]])
    #A = np.array([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], [3.0, 2.0, 0.0, 0.0, 1.0]])
    #A = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]])
    #A = np.array([[2., 1., 1., 0.], [4., 3., 3., 1.], [8., 7., 9., 5.], [6., 7., 9., 8.]])
    A = np.array([[3, 4, 2, 8, 5], [2, -1, 6, 10, 7], [5, 4, -3, 2, 0]])
    #A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    #b = np.array([[4.0], [6.0], [18.0]])
    #b = np.array([[1.], [2.], [4.], [5.]])
    b = np.array([[3], [5], [0]])
    #b = np.array([[3.], [11.], [17.]])

    dimA = np.shape(A)
    dimAexpandido = (dimA[0], dimA[1] + 1)
    Aexpandido = np.zeros(dimAexpandido)
    Aexpandido[0:, 0:dimA[1]] = A
    Aexpandido[0:, dimA[1]:] = b

    print("Aexpandido")
    print(Aexpandido)

    Aexpandido1 = np.copy(Aexpandido)

    #(matrizA, vetorIndices) = eliminacaoGauss(Aexpandido, 0.001)

    linhasIndices = [0, 1, 2]
    colunasIndices = [0, 1, 2, 3, 4]
    pivotInicial = 0
    numPivots = (np.shape(linhasIndices))[0]
    epsZero = 0.001
    (matrizA1, vetorIndices1, ehLD) = eliminacaoGaussSubMatriz(Aexpandido1, linhasIndices, colunasIndices, pivotInicial, numPivots, epsZero)

    #print("Aexpandido")
    #print(Aexpandido)
    #print("vetorIndices")
    #print(vetorIndices)
    #print("Matriz Reordenada")
    #print(reordenarMatriz(Aexpandido, vetorIndices))

    print("Aexpandido1")
    print(Aexpandido1)
    print("vetorIndices1")
    print(vetorIndices1)
    print("Matriz Reordenada1")
    print(reordenarMatriz(Aexpandido1, vetorIndices1))
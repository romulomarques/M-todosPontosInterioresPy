import numpy as np
from random import randint
from metodoTCInviavel import MetodoTCInviavel

#---------------------------- Geracao das instancias usando Distribuicao Uniform -------------------------

def gerarInstanciaUniforme(numLinhas, numColunas):
    # Numero de Linhas
    m = numLinhas

    # Numero de Colunas
    n = numColunas

    # Matriz de Restricoces 'A'
    temPostoCompleto = False
    while(temPostoCompleto == False):
        A = 100 * np.random.uniform(0.0, 1.0, (m, n))

        postoA = np.linalg.matrix_rank(A)
        if(postoA == np.amin(np.shape(A))):
            print("Posto de A")
            print(postoA)
            temPostoCompleto = True

    # Vetor lado direito 'b'
    b = 1 + 100 * np.random.uniform(0.0, 1.0, (m, 1))

    # Vetor e custos 'c'
    c = 1 + 100 * np.random.uniform(0.0, 1.0, (n, 1))

    return A, b, c

#----------------------------------------------------------------------------------------------------------

def gerarInstanciaNormal(numLinhas, numColunas):
    # Numero de Linhas
    m = numLinhas

    # Numero de Colunas
    n = numColunas

    # Matriz de Restricoces 'A'
    temPostoCompleto = False
    while(temPostoCompleto == False):
        A = np.random.normal(50.0, 35.0, (m, n))

        postoA = np.linalg.matrix_rank(A)
        if(postoA == np.amin(np.shape(A))):
            print("Posto de A")
            print(postoA)
            temPostoCompleto = True

    # Vetor lado direito 'b'
    b = np.random.normal(50.0, 35.0, (m, 1))

    # Vetor e custos 'c'
    c = np.random.normal(50.0, 35.0, (n, 1))

    return A, b, c
#----------------------------------------------------------------------------------------------------------

def geradorJefferson(numLinhas, numColunas):
    # Numero de Linhas
    m = numLinhas

    # Numero de Colunas
    n = numColunas

    # Matriz de Restricoces 'A'
    temPostoCompleto = False
    A = np.zeros((m,n))
    while (temPostoCompleto == False):
        indLinha = 0
        for linhaA in A:
            indColuna = 0
            for elemento in linhaA:
                linhaA[indColuna] = randint(0,9)
                indColuna = indColuna + 1
            indLinha = indLinha + 1

        postoA = np.linalg.matrix_rank(A)
        if (postoA == np.amin(np.shape(A))):
            print("Posto de A")
            print(postoA)
            temPostoCompleto = True

    # Vetor lado direito 'b'
    b = np.zeros((m, 1))
    indice = 0
    while(indice < m):
        b[indice, 0] = randint(3, 5)
        indice = indice + 1

    # Vetor e custos 'c'
    c = np.zeros((n, 1))
    indice = 0
    while(indice < n):
        c[indice, 0] = randint(3, 5)
        indice = indice + 1

    return A, b, c

def gerarInstanciasViavel(numLinhas, numColunas):
    identidade = np.identity(numLinhas)

    subMatrizAleatorio = np.zeros((numLinhas, numColunas - numLinhas))

    A = np.zeros((numLinhas, numColunas))

    # gerando pedaco aleatorio de A, A[0:, 0:numColunas-numLinhas].
    for linhaSubMatAleat in subMatrizAleatorio:
        indColuna = 0
        for elemento in linhaSubMatAleat:
            linhaSubMatAleat[indColuna] = randint(0, 9)
            indColuna = indColuna + 1

    # Montando matriz A como sendo: [matrizAleatoria(numLinhas, numColunas - numLinhas), identidade(numLinhas)]
    A[0:, 0:numColunas-numLinhas] = subMatrizAleatorio[:,:]
    A[0:, numColunas-numLinhas:] = identidade[:,:]

    # Vetor lado direito 'b'
    b = np.zeros((numLinhas, 1))
    indice = 0
    while (indice < numLinhas):
        b[indice, 0] = randint(3, 5)
        indice = indice + 1

    cAleatorio = np.zeros((numColunas-numLinhas, 1))
    indice = 0
    while (indice < numColunas-numLinhas):
        cAleatorio[indice, 0] = randint(6, 8)
        indice = indice + 1

    cFolga = np.zeros((numLinhas, 1))

    # Vetor de custos 'c'
    c = np.zeros((numColunas, 1))
    c[0:numColunas-numLinhas, 0] = cAleatorio[:,0]
    c[numColunas-numLinhas:, 0] = cFolga[:,0]

    return A, b, c

if(__name__ == "__main__"):
    numLinhas = 4
    numColunas = 12

    A, b, c = gerarInstanciasViavel(numLinhas, numColunas)

    print("A")
    print(A)
    print()

    print("b")
    print(b)
    print()

    print("c")
    print(c)
    print()
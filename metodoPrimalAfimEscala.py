# O metodo Primal Afim-Escala exige vetores na forma (tamanho,). Portanto, o vetor 'b', o vetor 'c' e o vetor 'x0'
# tem de ter as formas de arrays de dimensoes (m,), (n,) e (n,), respectivamente.

import numpy as np

def transformarParaArray1D(arrayCorrente):
    if (np.ndim(arrayCorrente) != 1):
        if (np.amin(np.shape(arrayCorrente)) != 1):
            print("Dimensao do ponto inicial estah incoerente.")
            return 0
        else:
            arrayReestruturado = np.reshape(arrayCorrente, np.size(arrayCorrente))
    else:
        arrayReestruturado = arrayCorrente

    return arrayReestruturado


class metodoPrimalAfimEscala:

    def __init__(self, A, b, c, x0, beta, valorDeParada):
        self._matrizA = A
        self._vetorBzinho = b
        self._vetorCzinho = c
        self._x0 = x0
        self._beta = beta
        self._valorDeParada = valorDeParada

        self._iteracoes = 0
        self._ultimoVetorXi = np.zeros(np.size(x0))

    def calcularMatrizProjecao(matrizA, dimN):
        # criando matriz identidade de dimensao 'n'
        identidadeN = np.eye(dimN)

        # criando matriz A*A_transposto
        matrizAAt = np.dot(matrizA, matrizA.transpose())

        # criando inversa da matriz A*A_transposto
        inversaAAt = np.linalg.inv(matrizAAt)

        # construindo matriz projecao
        matrizProjecaoA = np.dot(matrizA.transpose(), inversaAAt)
        matrizProjecaoA = np.dot(matrizProjecaoA, matrizA)
        matrizProjecaoA = identidadeN - matrizProjecaoA

        return matrizProjecaoA

    # Lembre que o vetor xi eh um ponto interior factivel, portanto todos os elementos de xi sao maiores
    # que zero. Portanto, se algum elemento 'j' do vetor 'd' eh negativo, a expressao (-xi(j)/d(j)) eh positiva.
    # Da mesma forma que (-xi(j)/d(j)) eh negativa quando d(j) eh positivo.

    # Agora, geramos o vetor com todas as divisoes (-xi(j)/ d(j)). Como o tamanho do passo 'alpha' eh:
    #                           min(em j){ (-xi(j)/ d(j)) | d(j) < 0 },
    # procuramos o menor valor do vetor de divisoes considerando somente os valores positivos.
    def calcularAlpha(esteVetorXi, esteVetorD):
        possiveisAlphas = np.divide(-esteVetorXi, esteVetorD)
        menorElemento = np.amax(possiveisAlphas)
        for elemento in possiveisAlphas:
            if (elemento > 0 and elemento < menorElemento):
                menorElemento = elemento
        if (menorElemento > 0):
            return menorElemento

    def aplicarMetodoPrimalAfimEscala(self, A, b, c, x0, beta, valorDeParada):
        iteracao = 0
        erroRelativo = valorDeParada + 1

        xi = x0

        while erroRelativo > valorDeParada:
            # setando qual iteracao eh essa
            iteracao = iteracao + 1

            # criando matriz diagonal Xi com diagonal principal igual aos elementos do vetor 'xi'.
            matrizXi = np.diag(xi)

            # criando matriz de restricao do problema afim, 'A_barra'. A_barra = A * Xi
            matrizBarraA = np.dot(A, matrizXi)

            # criando vetor de custos do problema afim, 'c_barra'. c_barra = Xi * c
            vetorBarraC = np.dot(matrizXi, c)

            # criando matriz de projecao de A_barra.
            matrizProjecaoBarraA = metodoPrimalAfimEscala.calcularMatrizProjecao(matrizBarraA, np.size(xi))

            # calculando direcao de melhora d_barra
            #vetorBarraD = - np.dot(matrizProjecaoBarraA, vetorBarraC)
            vetorBarraD = np.dot(matrizProjecaoBarraA, vetorBarraC)

            # transformando d_barra em d
            vetorD = np.dot(matrizXi, vetorBarraD)

            if all(elemento > 0 for elemento in vetorD):
                return "Problema Ilimitado"
            else:
                # calculando tamanho do passo alpha
                alpha = metodoPrimalAfimEscala.calcularAlpha(xi, vetorD)

                # guardando o ponto atual
                xi_menos_1 = xi

                # calculando novo ponto
                xi = xi + beta * alpha * vetorD

                # calculando erro relativo entre o novo ponto e o ponto antigo
                erroRelativo = (np.abs(np.dot(c, xi) - np.dot(c, xi_menos_1))) / np.abs(1 + np.dot(c, xi))

        return (xi, iteracao)

    def aplicarMetodoPrimalAfimEscalaEmMim(self):
        (self._ultimoVetorXi, self._iteracoes) = self.aplicarMetodoPrimalAfimEscala(self._matrizA, self._vetorBzinho, self._vetorCzinho, self._x0, self._beta, self._valorDeParada)

if (__name__ == "__main__"):
    A = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [3, 2, 0, 0, 1]])
    b = np.array([4, 6, 18])
    c = np.array([3, 5, 0, 0, 0])
    x0 = np.array([1, 1, 3, 5, 13])
    beta = 0.995
    valorDeParada = 0.0001

    resolvedorPrimalAfimEscala = metodoPrimalAfimEscala(A, b, c, x0, beta, valorDeParada)
    resolvedorPrimalAfimEscala.aplicarMetodoPrimalAfimEscalaEmMim()

    vetorSolucao = resolvedorPrimalAfimEscala._ultimoVetorXi
    numIt = resolvedorPrimalAfimEscala._iteracoes

    print("Valor Otimo:")
    print(np.dot(c, vetorSolucao))
    print("Numero de Iteracoes:")
    print(numIt)
    print("xi otimo:")
    print(vetorSolucao)

    B = np.eye(4)
    C = np.array([[2, 3], [4, 5]])

    B[1:3, 1:3] = C

    formaC = np.shape(C)
    formaD = (formaC[0], formaC[1] + 1)
    D = np.zeros(formaD)

    print("B:")
    print(B)
    print("C:")
    print(C)
    print("D:")
    print(D)

    print("cu")
    print(np.shape(b))
import numpy as np
from eliminacaoGauss import unitarizarDiag
from eliminacaoGauss import unitarizarDiagGaussMatrizQuadrada
from eliminacaoGauss import reordenarMatriz
from eliminacaoGauss import eliminacaoGaussSubMatriz
from copy import deepcopy

class MetodoTrajCentral:

    def __init__(self, matrizA, vetorBzinho, vetorCzinho, vetorXYSInicial, tal):
        self._matrizA = matrizA
        self._vetorBzinho = vetorBzinho
        self._vetorCzinho = vetorCzinho
        self._vetorXYSInicial = vetorXYSInicial
        self._tal = tal

        self._matrizX = self.montarMatrizX(self._vetorXYSInicial, np.shape(matrizA))
        self._matrizS = self.montarMatrizS(self._vetorXYSInicial, np.shape(matrizA))
        self._matrizI = self. montarMatrizI(np.shape(matrizA))

    # dimensoesMatrizA: Tupla que guarda a quantidade de linhas e colunas, nesta ordem,  da matriz A.
    def localizarVariaveisXYS(self, dimensoesMatrizA):
        # Como nosso vetor de variaveis eh formado pela concatenacao das variaveis x, y e s nesta ordem, decidimos guardar
        # os indices de onde comeca e onde termina cada tipo de variavel. Desta maneira, consiguiremos acessar rapidamente
        # as variaveis do vetor 'xys' que queremos manipular no momento.
        comecoEfimX = (0, dimensoesMatrizA[1] - 1)
        comecoEfimY = (dimensoesMatrizA[1], dimensoesMatrizA[1] + dimensoesMatrizA[0] - 1)
        comecoEfimS = (dimensoesMatrizA[1] + dimensoesMatrizA[0], 2*dimensoesMatrizA[1] + dimensoesMatrizA[0] - 1)

        return (comecoEfimX, comecoEfimY, comecoEfimS)

    # Adicionar colunas no fim de uma matriz. Tais colunas correspondem ao parametro matrixExpansao.
    def expandirMatriz(self, matrizOriginal, matrizExpansao):
        dimMatrizOriginal = np.shape(matrizOriginal)
        dimMatrizExpansao = np.shape(matrizExpansao)

        dimMatrizExpandida = (dimMatrizOriginal[0], dimMatrizOriginal[1] + dimMatrizExpansao[1])

        matrizExpandida = np.zeros(dimMatrizExpandida)

        # Fazendo bloco (0:m-1, 1:n-1) da matriz expandida ser igual a matriz original.
        matrizExpandida[0:, 0:dimMatrizOriginal[1]] = matrizOriginal

        # Fazendo as ultimas colunas da matriz expandida, bloco (0:m-1, n:n+n'), onde (m,n) eh a dimensao da matriz
        # original e n' eh o numero de colunas novas a serem integradas, receber a matriz expansao, a matriz
        # correspondida ahs colunas novas.
        matrizExpandida[0:dimMatrizExpandida[0], dimMatrizOriginal[1]:] = matrizExpansao

        return matrizExpandida

    def montarMatrizX(self, vetorXYS, dimA):
        # Conseguindo os indices de inicio e fim das variaveis do tipo 'x' (vetor de variaveis primais), 'y' (vetor de variaveis duais)
        # e 's' (vetor de variavies de folga do problema dual) que compoem o vetor vetorXYS.
        (comecoEfimX, comecoEfimY, comecoEfimS) = self.localizarVariaveisXYS(dimA)

        # Capturando o vetor de variaveis primais 'x' de dentro do vetor vetorXYS.
        vetorX = vetorXYS[comecoEfimX[0]: comecoEfimX[1] + 1]

        # Lembre que nossos vetores sao definidos da seguinte maneira: ('l' linhas, 1 coluna). Uma vez que o comando 'diag' precisa de
        # um array do tipo ('l' linhas) para retornar a matriz de diagonal associada ao vetor de variaveis primais 'x', nos usamos o
        # comando vetorX[:, 0] para fazer o metodo 'diag' funcionar corretamente.
        matrizX = np.diag(vetorX[:, 0])

        return matrizX

    def montarMatrizS(self, vetorXYS, dimA):
        # Conseguindo os indices de inicio e fim das variaveis do tipo 'x' (vetor de variaveis primais), 'y' (vetor de variaveis duais)
        # e 's' (vetor de variavies de folga do problema dual) que compoem o vetor vetorXYS.
        (comecoEfimX, comecoEfimY, comecoEfimS) = self.localizarVariaveisXYS(dimA)

        # Capturando o vetor de variaveis de folga 's' de dentro do vetor vetorXYS.
        vetorS = vetorXYS[comecoEfimS[0]: comecoEfimS[1] + 1]

        # Criando matriz diagonal associada ao vetor de variaveis de folga 's'. Tal processo eh identico a criacao da matriz diagonal
        # associada ao vetor de variaveis primais 'x'
        matrizS = np.diag(vetorS[:, 0])

        return matrizS

    def montarMatrizI(self, dimA):
        # Criando matriz identidade 'n por n', onde 'n' eh o numero de variaveis 's'. A matriz '-matrizI' corresponde ah derivada parcial
        # de 'At*y -s -c' com respeito ah 's'.
        matrizI = np.eye(dimA[1])

        return matrizI

    # Jacobiana pode ser entendido como o 'gradiente' de uma funcao F_generica: R(n)->R(n).No nosso caso, usaremos a Jacobiana
    # da funcao F do R(2*n + m) no R(2*n + m), que representa o nosso espaco de solucoes. Temos F:
    #                    [ At*y - s - c  ]
    #       F(x, y, s) = [ A*x - b     ]
    #                    [ X*S*e - u*e ]
    # Sua Jacobiana eh dada por:
    #                            [ 0  At  -I ]
    #       Jacobiana(x, y, s) = [ A   0   0 ]
    #                            [ S   0   X ]

    def montarRohB(self, matrizA, vetorB, primalXi):
        # Vetor RohB = b - A*xi.
        # Note que RohB representa folgas existentes que distanciam o vetor de variaveis primais 'xi' de ser primal viavel.
        rohB = vetorB - np.dot(matrizA, primalXi)

        return rohB

    def montarRohC(self, matrizA, vetorC, dualYi, folgaDualSi):
        # Vetor RohC = c - At*yi + si.
        # Note que RohC representa folgas existentes que distanciam os vetores de variaveis duais 'yi' e folgas do dual 'si',
        # de constituirem solucao dual viavel.
        rohC = vetorC - np.dot(matrizA.transpose(), dualYi) + folgaDualSi

        return rohC

    # Neste metodo, montaremos a matriz Jacobiana acima.
    def montarJacobiana(self, matrizA, matrizX, matrizS, matrizI):
        # Dimensoes da matriz A.
        dimA = np.shape(matrizA)

        # Dimensao da matriz jacobiana: (2*n + m) por (2*n + m), onde 'n' eh o numero de colunas de A e 'm' eh o numero
        # de linhas de A.
        dimJacob = (2 * dimA[1] + dimA[0], 2 * dimA[1] + dimA[0])

        # Criando matriz jacobiana toda zerada.
        jacobiana = np.zeros(dimJacob)

        # Setando bloco matricial de posicao [0, 1]: At.
        jacobiana[0:dimA[1], dimA[1]:dimA[1]+dimA[0]] = matrizA.transpose()

        # Setando bloco matricial de posicao [0, 2]: -I.
        jacobiana[0:dimA[1], dimA[1]+dimA[0]:] = -matrizI

        # Setando bloco matricial de posicao [1, 0]: A.
        jacobiana[dimA[1]:dimA[1]+dimA[0], 0: dimA[1]] = matrizA

        # Setando bloco matricial de posicao [2, 0]: S.
        jacobiana[dimA[1]+dimA[0]:, 0:dimA[1]] = matrizS

        # Setando bloco matricial de posicao [2, 2]: X.
        jacobiana[dimA[1]+dimA[0]:, dimA[1]+dimA[0]:] = matrizX

        return jacobiana

    def montarMinhaJacobiana(self):
        return self.montarJacobiana(self._matrizA, self._matrizX, self._matrizS, self._matrizI)

    def atualizarMatrizX(self, matrizX, novoVetorX):
        dimMatrizX = np.shape(matrizX)
        dimNovoVetorX = np.shape(novoVetorX)

        if(dimMatrizX[0] == dimNovoVetorX[0]):
            cont = 0
            for elemento in novoVetorX:
                matrizX[cont, cont] = elemento
                cont = cont + 1
        else:
            print("O novo vetor X NAO possui quantidade de elementos igual ao numero de elementos na diagonal principal da matriz X.")

        return matrizX

    def atualizarMatrizS(self, matrizS, novoVetorS):
        dimMatrizS = np.shape(matrizS)
        dimNovoVetorS = np.shape(novoVetorS)

        if(dimMatrizS[0] == dimNovoVetorS[0]):
            cont = 0
            for elemento in novoVetorS:
                matrizS[cont, cont] = elemento
                cont = cont + 1
        else:
            print("O novo vetor S NAO possui quantidade de elementos igual ao numero de elementos na diagonal principal da matriz S.")

        return matrizS

    def atualizarMatrizJacobiana(self, jacobiana, primalXi, folgaDualSi, dimA):
        dimPrimalXi = np.shape(primalXi)
        dimFolgaDualSi = np.shape(folgaDualSi)

        # Atualizando bloco matricial de posicao [2, 0]: S.
        if (dimFolgaDualSi[0] == dimA[1]):
            contJacLinha = dimA[1] + dimA[0]
            contJacColuna = 0

            for elemento in folgaDualSi:
                jacobiana[contJacLinha, contJacColuna] = float(elemento)
                contJacLinha = contJacLinha + 1
                contJacColuna = contJacColuna + 1
        else:
            print("NAO mudou matriz S")

        # Atualizando bloco matricial de posicao [2, 2]: X.
        if (dimPrimalXi[0] == dimA[1]):
            contJacLinha = dimA[1] + dimA[0]
            contJacColuna = dimA[1] + dimA[0]

            for elemento in primalXi:
                jacobiana[contJacLinha, contJacColuna] = float(elemento)
                contJacLinha = contJacLinha + 1
                contJacColuna = contJacColuna + 1
        else:
            print("NAO mudou matriz X")

        return jacobiana

    def calcularPesoMi(self, primalXi, folgaDualSi):
        # mi = (xi*si)/n, onde 'n' eh o numero de linhas do vetor 'xi', que eh igual a dimensao do vetor 'si'.
        if(np.shape(primalXi)[0] == np.shape(folgaDualSi)[0]):
            mi = (np.dot(primalXi.transpose(), folgaDualSi))/(np.shape(primalXi)[0])
        else:
            mi = -1
            print("O numero de linhas do vetor primalXi NAO eh igual ao numero de linhas do vetor folgaDualSi.")

        # Quando usamos a funcao np.dot() em dois vetores, o resultado eh um array de uma coluna e uma linha. Aqui, estamos
        # retornando somente o valor numerico de 'mi' calculado, e nao este valor dentro de um array (1, 1).
        return float(mi)

    # O trecho Mi é:
    #       [ pesoMi*e - X*S*e ]
    #
    # Do vetor:
    #       [       rohC       ]
    #       [       rohB       ]
    #       [ pesoMi*e - X*S*e ]
    def calcularTrechoMi(self, primalXi, folgaDualSi, pesoMi, tal):
        trechoMiParte1 = np.ones(np.shape(primalXi))
        trechoMiParte1 = tal * pesoMi * trechoMiParte1

        trechoMiParte2 = np.multiply(primalXi, folgaDualSi)

        trechoMi = trechoMiParte1 - trechoMiParte2

        return trechoMi

    def calcularMelhorPasso(self, vetorNumerador, vetorDenominador):
        possiveisPassos = np.divide(-vetorNumerador, vetorDenominador)
        menorElemento = np.amax(possiveisPassos)

        for elemento in possiveisPassos:
            if (elemento > 0 and elemento < menorElemento):
                menorElemento = elemento

        return float(menorElemento)

    def calcularGapPrimalDual(self, primalXi, dualYi, vetorBzinho, vetorCzinho):
        gap = np.abs(np.dot(vetorBzinho.transpose(), dualYi) - np.dot(vetorCzinho.transpose(), primalXi))
        return  float(gap)

    # Precisamos resolver o seguinte sistema de equacoes para achar a soulcao primal-dual viavel com respeito ao nosso 'u' atual:
    #       [ 0  At  -I ]     [deltaX]     [ 0               ]
    #       [ A   0   0 ]  *  [deltaY]  =  [ 0               ]
    #       [ S   0   X ]     [deltaS]     [ tal*u*e - X*S*e ]

    # Chamamos o vetor correspondente ao lado direito do sistema linear em questao de vetorMi. Ou seja:
    #                 [ 0               ]
    #       vetorMi = [ 0               ]
    #                 [ tal*u*e - X*S*e ]

    # Neste metodo, calcularemos o vetor [deltaX, deltaY, deltaS] transposto (em peh).
    def calcularDeltaXYS(self, jacobiana, vetorMi, dimA, epsZero):
        # Printando os valores do tipo float com duas casas decimais de precisao.
        np.set_printoptions(precision=2)

        # Adicionando a coluna composta por 'vetorMi' no fim da matriz jacobiana.
        # Entao, nossa Matriz Jacobiana Expandida atual eh:
        #       [ 0   At  -I   vetorMi(linha 0) ]
        #       [ A   0  -0    vetorMi(linha 1) ]
        #       [ S   0    X   vetorMi(linha 2) ]
        jacobianaExpandida = self.expandirMatriz(jacobiana, vetorMi)

        # Dimensao da Jacobiana expandida.
        dimJacExp = np.shape(jacobianaExpandida)

        # Criando vetor que indica a posicao que as linhas estao ocupando na matriz.
        linhasIndices = np.arange(dimJacExp[0])

        # Setando o bloco [ A  0  0  vetorMi(linha 1) ] como primeiro bloco e o bloco [ 0  At  -I  vetorMi(linha 0) ] como segundo bloco.
        troca = np.copy(linhasIndices[0:dimA[1]])
        linhasIndices[0:dimA[0]] = linhasIndices[dimA[1]:dimA[1]+dimA[0]]
        linhasIndices[dimA[0]:dimA[1]+dimA[0]] = troca

        # Neste ponto, nossa matriz jacobiana expandida estah:
        #       [ A   0    0    vetorMi(linha 0) ]
        #       [ 0   At  -I    vetorMi(linha 1) ]
        #       [ S   0    X    vetorMi(linha 2) ]
        # Portanto, primeiramente aplicaremos Gauss nas 'n' primeiras colunas. Neste processo, modificaremos
        # somente as 'n' primeiras colunas e nas ultimas 'n+1' colunas.

        # Colunas a serem modificadas na aplicacao de Gauss nas 'n' primeiras colunas.
        colunasIndices = np.arange(2 * dimA[1] + 1)
        colunasIndices[dimA[1]:2 * dimA[1] + 1] = np.arange(dimA[1] + dimA[0], 2 * dimA[1] + dimA[0] + 1)[:]

        # Indice do primeiro pivot da aplicacao de Gauss que estah por vir.
        pivotInicial = 0

        numPivots = dimA[1]

        (jacobianaExpandida, linhasIndices, ehLD) = eliminacaoGaussSubMatriz(jacobianaExpandida, linhasIndices, colunasIndices, pivotInicial, numPivots, epsZero)

        # Se a Jacobiana expandida nao eh LD, aplicamos Eliminacao de Gauss nas 'n+m' colunas restantes.
        if(ehLD is True):
            deltaXYS = np.zeros((dimJacExp[0], 1))
        else:
            colunasIndices = np.arange(dimA[1], 2*dimA[1] + dimA[0] + 1)
            pivotInicial = dimA[1]
            numPivots = (2*dimA[1] + dimA[0]) - (dimA[1])
            (jacobianaExpandida, linhasIndices, ehLD) = eliminacaoGaussSubMatriz(jacobianaExpandida, linhasIndices, colunasIndices, pivotInicial, numPivots, epsZero)

            if(ehLD is True):
                deltaXYS = np.zeros((dimJacExp[0], 1))
            else:
                unitarizarDiagGaussMatrizQuadrada(jacobianaExpandida, linhasIndices)
                jacobianaExpReord = reordenarMatriz(jacobianaExpandida, linhasIndices)

                # Capturando o vetor 'delta' como um vetor coluna.
                deltaXYS = np.zeros((dimJacExp[0], 1))
                deltaXYS[:, 0] = jacobianaExpReord[:, -1]

        return deltaXYS, ehLD

if(__name__ == "__main__"):
    A = np.array([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], [3.0, 2.0, 0.0, 0.0, 1.0]])
    b = np.array([[4.0], [6.0], [18.0]])
    c = np.array([[-3.0], [-5.0], [0.0], [0.0], [0.0]])
    x0 = np.array([[1.0], [1.0], [3.0], [5.0], [13.0]])
    tal = 0.4
    valorDeParada = 0.0001

    vetorXYS = np.zeros((13,1))
    vetorXYS[0 : 5] = x0
    vetorXYS[5 : 8] = np.array([[7], [11], [15]])
    vetorXYS[8 : 13] = np.array([[9], [14], [17], [21], [23]])

    trajCentral = MetodoTrajCentral(A, b, c, vetorXYS, tal)

    (comecoEfimX, comecoEfimY, comecoEfimS) = trajCentral.localizarVariaveisXYS(np.shape(A))

    jacobiana = trajCentral.montarMinhaJacobiana()

    # Nossa representacao numerica do zero.
    epsZero = 0.0

    vetorMi = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    deltaXYS, ehLD = trajCentral.calcularDeltaXYS(jacobiana, vetorMi, np.shape(A), epsZero)

    print("vetorMi")
    print(vetorMi)
    print("deltaXYS")
    print(deltaXYS)
    print("jacobiana * deltaXYS")
    print(np.dot(jacobiana, deltaXYS))

    listaA = np.array([[1.], [3.], [5.]])
    listaB = np.array([[2.], [4.], [6.]])
    print("multiplicacao Listas")
    listaC = np.multiply(listaA, listaB)
    print(listaC)

    listaD = np.array([[1.], [7.], [-1.5], [-2.7], [3.2], [4.]])
    print("menor valor da listaD")
    print(np.amin(listaD))

    #primalXi = np.array([[-5], [-5], [-5], [-5], [-5]])
    #folgaDualSi = np.array([[-2], [-2], [-2], [-2], [-2]])

    #trajCentral.atualizarMatrizJacobiana(jacobiana, primalXi, folgaDualSi, dimA = np.shape(A))

    #print("jacLele")
    #print(jacobiana)
import numpy as np
import time
from lerArquivoMP import lerInstanciaMPIFormato1
from metodoPrimalAfimEscala import metodoPrimalAfimEscala
from metodoPrimalAfimEscala import transformarParaArray1D

class metodoDuasFases:

    def __init__(self, A, b, c, x0, beta, valorDeParada):
        self._matrizA = metodoDuasFases.montarRestricoes(self, A, b, x0)
        self._vetorBzinho = b
        self._vetorCzinho = metodoDuasFases.montarVetorDeCustos(self, c)
        self._x0 = metodoDuasFases.montarVetorDeVariaveis(self, x0)

        self._resolvedor = metodoPrimalAfimEscala(self._matrizA, self._vetorBzinho, self._vetorCzinho, self._x0, beta, valorDeParada)

    # Supondo que a matriz A tem dimensao (m, n), a matriz de restricoes 'matrizRestricoes' do problema
    # duas fazes terah dimensao (m, n+1), onde o bloco (1:m, 1:n) de matrizRestricoes corresponde ah matriz
    # A e a ultima coluna de matrizRestricoes, bloco (1:m, n+1), corresponde aos coeficientes da variavel 'u'.
    def montarRestricoes(self, matrizA, vetorBzinho, x0):
        shapeA = np.shape(matrizA)
        shapeEstaMatrizRestricoes = (shapeA[0], shapeA[1] + 1)
        estaMatrizRestricoes = np.zeros(shapeEstaMatrizRestricoes)

        # Fazendo bloco (1:m, 1:n) de matrizRestricoes igual a matriz A
        estaMatrizRestricoes[0:shapeA[0], 0:shapeA[1]] = matrizA

        # coeficientes da variavel 'u': (b - A*x0)
        # Note que este array é de fato um vetor. Tem 'm' espacos.
        coeficientesUzinho = vetorBzinho - np.dot(matrizA, x0)

        # tornando o vetor de coeficientes de 'u' em uma coluna de maatriz de forma (m, 1).
        matrizVetorUzinho = coeficientesUzinho.reshape((np.size(coeficientesUzinho), 1))

        # Fazendo a ultima coluna de matrizRestricoes, bloco (1:m, n+1), igual a matriz (m, 1) dos coeficientes
        # de 'u'
        estaMatrizRestricoes[0:shapeEstaMatrizRestricoes[0], shapeEstaMatrizRestricoes[1]-1:] = matrizVetorUzinho

        return estaMatrizRestricoes

    def montarVetorDeCustos(self, vetorCzinho):
        tamNovoVetorCzinho = np.size(vetorCzinho) + 1
        novoVetorCzinho = np.zeros((tamNovoVetorCzinho))

        # setando ultima posicao do novo vetor de custos igual a '-1', pois este eh o coeficiente de custo
        # da variavel 'u'.
        novoVetorCzinho[-1] = -1

        return novoVetorCzinho

    def montarVetorDeVariaveis(self, x0):
        tamNovoVetorX0 = np.size(x0)

        novoVetorX0 = np.zeros(tamNovoVetorX0 + 1)

        # Setando as 'n' primeiras posicaoes do vetor da solucao inicial como sendo o valor nicial
        # das variaveis 'x'
        novoVetorX0[0:tamNovoVetorX0] = x0

        # Setando o valor inicial da variavel 'u' no vetor da solucao inicial, ou seja, setando a ultima posicao
        # do do vetor da solucao inicial. Setamos seu valor igual a 1.
        novoVetorX0[np.size(x0):] = 1

        return novoVetorX0

    # Acha um ponto x0 inicial que eh interior factivel para o problema original.
    def calcularPontoInteriorFactivel(self):
        self._resolvedor.aplicarMetodoPrimalAfimEscalaEmMim()

if(__name__ == "__main__"):
    nomeArquivo = "Instancias\\instancia8.txt"
    m, n, c, b, A = lerInstanciaMPIFormato1(nomeArquivo)
    x0 = np.ones((n,1))
    beta = 0.995
    valorDeParada = 1E-5

    x0 = transformarParaArray1D(x0)
    b = transformarParaArray1D(b)
    c = transformarParaArray1D(c)

    tempoTotal = 0

    objetoMetodoDuasFases = metodoDuasFases(A, b, c, x0, beta, valorDeParada)

    tempoInicio = time.process_time()
    objetoMetodoDuasFases.calcularPontoInteriorFactivel()
    tempoFinal = time.process_time()

    tempoTotal = tempoTotal + (tempoFinal - tempoInicio)

    solucaoOtimaPrimeiraFase = objetoMetodoDuasFases._resolvedor._ultimoVetorXi
    vetorCzinho = objetoMetodoDuasFases._resolvedor._vetorCzinho
    print("solucaoOtimaPrimeiraFase")
    print(solucaoOtimaPrimeiraFase)
    print("valorOtimoPrimeiraFase")
    print(np.dot(solucaoOtimaPrimeiraFase, vetorCzinho))
    print("A*solucaoOtimaPrimeiraFase:")
    print(np.dot(objetoMetodoDuasFases._matrizA, solucaoOtimaPrimeiraFase))

    tempoInicio = time.process_time()
    (solucaoOtimaSegundaFase, iteracoesSegundaFase) = objetoMetodoDuasFases._resolvedor.aplicarMetodoPrimalAfimEscala(A, b, c, solucaoOtimaPrimeiraFase[0:-1], beta, valorDeParada)
    tempoFinal = time.process_time()

    tempoTotal = tempoTotal + (tempoFinal - tempoInicio)
    print("Valor Otimo Segunda Fase")
    print(np.dot(c.transpose(), solucaoOtimaSegundaFase))
    print("Solucao Otima Segunda Fase")

    print("Tempo Total")
    print(tempoTotal)

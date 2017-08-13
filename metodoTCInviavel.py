# Todos os vetores tem de ser criados verticalmente, ou seja, a dimensao dos vetores tem de ser do tipo (n,1)

import time
import numpy as np
from metodosTrajCentral import MetodoTrajCentral
from lerArquivoMP import lerInstanciaMPIFormato1

class MetodoTCInviavel:

    def __init__(self, A, b, c, vetorXYSInicial, tal, valorDeParada, alpha = 0.8, beta = 1.4):
        self._meuMetodoTC = MetodoTrajCentral(A, b, c, vetorXYSInicial, tal)
        self._valorDeParada = valorDeParada
        self._alpha = alpha
        self._beta = beta

        (comecoEfimX, comecoEfimY, comecoEfimS) = self._meuMetodoTC.localizarVariaveisXYS(np.shape(A))
        self._rohBinicial = self._meuMetodoTC.montarRohB(A, b, vetorXYSInicial[comecoEfimX[0]: comecoEfimX[1]+1])
        self._rohCinicial = self._meuMetodoTC.montarRohC(A, c, vetorXYSInicial[comecoEfimY[0]: comecoEfimY[1]+1], vetorXYSInicial[comecoEfimS[0]: comecoEfimS[1]+1])
        self._pesoMiInicial = self._meuMetodoTC.calcularPesoMi(vetorXYSInicial[comecoEfimX[0]: comecoEfimX[1]+1], vetorXYSInicial[comecoEfimS[0]: comecoEfimS[1]+1])

        rohCRohBInicial = np.ones((np.shape(self._rohCinicial)[0] + np.shape(self._rohBinicial)[0], 1))
        rohCRohBInicial[0:np.shape(self._rohCinicial)[0], 0] = self._rohCinicial[:, 0]
        rohCRohBInicial[np.shape(self._rohCinicial)[0]:np.shape(self._rohCinicial)[0]+np.shape(self._rohBinicial)[0], 0] = self._rohBinicial[:, 0]

        self._normaRohCRohBInicial = np.linalg.norm(rohCRohBInicial, np.inf)

    def distanciaMenosInfTC(self, primalXi, folgaDualSi):
        # Calculando peso Mi. Temos que: Mi = (x_transposto * s)/n.
        pesoMi = MetodoTrajCentral.calcularPesoMi(self, primalXi, folgaDualSi)

        if(pesoMi >= 0):
            # Calculando o vetor [ xi[0]*si[0], xi[1]*si[1], xi[2]*si[2], ..., xi[n-1]*si[n-1] ].
            vetorXSe = np.multiply(primalXi, folgaDualSi)
            # Calculando o vetor de tamanho 'n' [ mi, mi, mi, ...., mi ].
            vetorMiE = pesoMi * np.ones((np.shape(primalXi)[0], 1))

            # Vetor que representa a distancia da solucao atual para a trajetoria central.
            # Tal vetor tem a forma: [ xi[0]*si[0] - mi, xi[1]*si[1] - mi, ....., xi[n-1]*si[n-1] - mi ].
            conteudoNorma = vetorXSe - vetorMiE

            # Calculando a norma menos infinito da distancia. Lembre que a norma menos infinito corresponde ah menor distancia
            # de uma coordenada para a origem. Portanto, capturamos o menor modulo de coordenada.
            #normaMenosInf = np.amin(np.abs(conteudoNorma))
            normaMenosInf = np.linalg.norm(conteudoNorma, -np.inf)

            # Especie de normalizacao da norma pelo peso Mi.
            distanciaTC = float(normaMenosInf / pesoMi)
        else:
            distanciaTC = -1
            print("NAO foi possivel calcular o pesoMi para o calculo da distancia.")

        return distanciaTC

    def pertenceAhVizinhanca(self, primalXi, folgaDualSi, rohB, rohC):
        # Calculando distancia da solucao atual para a trajetoria central.
        normaDelta = self.distanciaMenosInfTC(primalXi, folgaDualSi)
        if(normaDelta < 0):
            normaDelta = self._alpha + 1

        # Condicao 1: A distancia da solucao atual para a TC eh menor ou igual ao parametro alpha?
        satisfazDistancia = (normaDelta <= self._alpha)

        # Montando vetor [rohC, rohB]
        rohCRohB = np.ones((np.shape(rohC)[0] + np.shape(rohB)[0], 1))
        rohCRohB[0:np.shape(rohC)[0], 0] = rohC[:, 0]
        rohCRohB[np.shape(rohC)[0]:np.shape(rohC)[0] + np.shape(rohB)[0], 0] = rohB[:, 0]

        # Calculando Norma do vetor [rohC, rohB]
        #normaRohCRohB = np.amin(np.abs(rohCRohB))
        normaRohCRohB = np.linalg.norm(rohCRohB, -np.inf)

        # Calculando Mi = (x_transposto * s)/n, onde 'n' eh o tamanho dos vetores 'x' e 's'.
        pesoMi = self._meuMetodoTC.calcularPesoMi(primalXi, folgaDualSi)

        # Condicao 2: A Relacao das normas dos erros Rohs estah marjorada por um multiplo do parametro beta?
        satisfazNormasRohs = ( (normaRohCRohB / self._normaRohCRohBInicial) <= (self._beta * (pesoMi / self._pesoMiInicial)) )

        # Condicao 3: Todas as coordenadas do vetor de variaveis primais 'x' sao estritamente positivas?
        satisfazPositivoXi = all(elemento > 1E-6 for elemento in primalXi)

        # Condicao 4: Todas as coordenadas do vetor das folgas duais 's' sao estritamente positivas?
        satisfazPositivoSi = all(elemento > 1E-6 for elemento in folgaDualSi)

        # Todas as condicoes anteriores foram satisfeitas?
        #   Se sim, entao a solucao atual estah na vizinhanca;
        #   Se nao, entao a solucao atual NAO estah na vizinhanca.
        satisfazVizinhanca = satisfazDistancia and satisfazNormasRohs and satisfazPositivoXi and satisfazPositivoSi

        return satisfazVizinhanca

    def calcularPassoVizinhanca(self, passoLambda, varXYS, varDeltaXYS):
        pertenceVizinhanca = False
        passoEhMuitoPequeno = False
        diminuidorPasso = 1
        
        while(pertenceVizinhanca is False and passoEhMuitoPequeno is False):
            diminuidorPasso = diminuidorPasso * 0.95
            proximoPontoXYS = varXYS + diminuidorPasso * passoLambda * varDeltaXYS

            (comecoEfimX, comecoEfimY, comecoEfimS) = self._meuMetodoTC.localizarVariaveisXYS(np.shape(self._meuMetodoTC._matrizA))

            primalXi = proximoPontoXYS[comecoEfimX[0]:comecoEfimX[1]+1]
            dualYi = proximoPontoXYS[comecoEfimY[0]:comecoEfimY[1]+1]
            folgaDualSi = proximoPontoXYS[comecoEfimS[0]:comecoEfimS[1]+1]

            rohB = self._meuMetodoTC.montarRohB(self._meuMetodoTC._matrizA, self._meuMetodoTC._vetorBzinho, primalXi)
            rohC = self._meuMetodoTC.montarRohC(self._meuMetodoTC._matrizA, self._meuMetodoTC._vetorCzinho, dualYi, folgaDualSi)

            pertenceVizinhanca = self.pertenceAhVizinhanca(primalXi, folgaDualSi, rohB, rohC)

            if (diminuidorPasso < 0.001):
                passoEhMuitoPequeno = True

        return float(diminuidorPasso * passoLambda), passoEhMuitoPequeno

    def aplicarMetodoTCInviavel(self, epsZero):
        iteracao = 0
        tal = np.copy(self._meuMetodoTC._tal)
        dimA = np.shape(self._meuMetodoTC._matrizA)
        vetorXYSCorrente = np.copy(self._meuMetodoTC._vetorXYSInicial)
        melhorSolucaoXYS = np.copy(vetorXYSCorrente)

        (comecoEfimX, comecoEfimY, comecoEfimS) = self._meuMetodoTC.localizarVariaveisXYS(dimA)

        jacobiana = self._meuMetodoTC.montarMinhaJacobiana()
        vetorMiCorrente = np.zeros((np.shape(jacobiana)[0], 1))

        procuraPontoMelhor = True

        while(procuraPontoMelhor is True):
            print("iteracao")
            print(iteracao)

            primalXi = vetorXYSCorrente[comecoEfimX[0]: comecoEfimX[1]+1]
            dualYi = vetorXYSCorrente[comecoEfimY[0]: comecoEfimY[1]+1]
            folgaDualSi = vetorXYSCorrente[comecoEfimS[0]: comecoEfimS[1]+1]

            pesoMiCorrente = self._meuMetodoTC.calcularPesoMi(primalXi, folgaDualSi)

            rohBCorrente = self._meuMetodoTC.montarRohB(self._meuMetodoTC._matrizA, self._meuMetodoTC._vetorBzinho, primalXi)
            rohCCorrente = self._meuMetodoTC.montarRohC(self._meuMetodoTC._matrizA, self._meuMetodoTC._vetorCzinho, dualYi, folgaDualSi)

            # Montando vetorMi:
            #                 [          rohC          ]
            #       vetorMi = [          rohB          ]
            #                 [  tal*pesoMi*e - X*S*e  ]
            vetorMiCorrente[0:dimA[1]] = rohCCorrente
            vetorMiCorrente[dimA[1]:dimA[1]+dimA[0]] = rohBCorrente
            vetorMiCorrente[dimA[1]+dimA[0]:] = self._meuMetodoTC.calcularTrechoMi(primalXi, folgaDualSi, pesoMiCorrente, tal)

            # Calculando vetor deltaXYS = [deltaXi, deltaYi, deltaSi]_transposto tal que:
            #       [ 0  At  -I ]     [deltaXi]     [       rohC      ]
            #       [ A   0   0 ]  *  [deltaYi]  =  [       rohB      ]
            #       [ S   0   X ]     [deltaSi]     [ tal*u*e - X*S*e ]
            deltaXYS, ehLD = self._meuMetodoTC.calcularDeltaXYS(jacobiana, vetorMiCorrente, dimA, 1E-6)

            atualGap = self._meuMetodoTC.calcularGapPrimalDual(primalXi, dualYi, self._meuMetodoTC._vetorBzinho,
                                                               self._meuMetodoTC._vetorCzinho)
            # 1------------------------------------------------
            print("esteGap")
            print(atualGap)
            # 1------------------------------------------------

            if(ehLD is False):
                # Calculando melhor passo com respeito as variaveis 'xi'.
                passoXi = self._meuMetodoTC.calcularMelhorPasso(primalXi, deltaXYS[comecoEfimX[0]: comecoEfimX[1] + 1])

                # Calculando melhor passo com respeito as variaveis 'si'.
                passoSi = self._meuMetodoTC.calcularMelhorPasso(folgaDualSi,
                                                                deltaXYS[comecoEfimS[0]: comecoEfimS[1] + 1])

                # Capturando o menor passo entre os passos com respeito a 'xi' e a 'si'.
                passoMenor = np.amin((passoXi, passoSi))

                print("passoMenor")
                print(passoMenor)

                if(passoMenor > 0):
                    # Calculando multiplo dde 'passoMenor' que mantem o proximo ponto (x',y',s') na vizinhanca.
                    passoVizinhanca, passoMelhorNaoEncontrado = self.calcularPassoVizinhanca(passoMenor, vetorXYSCorrente, deltaXYS)

                    if(passoMelhorNaoEncontrado == False):
                        # Calculando proximo ponto (x',y',s').
                        novoPonto = vetorXYSCorrente + passoVizinhanca * deltaXYS

                        # Capturando as variaveis primais 'x' e as variaveis de folga duais 's' do novo vetor de variaveis XYS.
                        novoPrimalXi = novoPonto[comecoEfimX[0]: comecoEfimX[1] + 1]
                        novoDualYi = novoPonto[comecoEfimY[0]: comecoEfimY[1] + 1]
                        novoFolgaDualSi = novoPonto[comecoEfimS[0]: comecoEfimS[1] + 1]

                        novoGap = self._meuMetodoTC.calcularGapPrimalDual(novoPrimalXi, novoDualYi,
                                                                            self._meuMetodoTC._vetorBzinho,
                                                                            self._meuMetodoTC._vetorCzinho)

                        print("novoGap")
                        print(novoGap)

                        if(novoGap > atualGap):
                            rohCRohB = np.ones((np.shape(rohCCorrente)[0] + np.shape(rohBCorrente)[0], 1))
                            rohCRohB[0:np.shape(rohCCorrente)[0], 0] = rohCCorrente[:, 0]
                            rohCRohB[np.shape(rohCCorrente)[0]:np.shape(rohCCorrente)[0] + np.shape(rohBCorrente)[0], 0] = rohBCorrente[:, 0]
                            self._normaRohCRohBInicial = np.linalg.norm(rohCRohB, np.inf)

                        # Atualizando o pesoMi para o pesoMi do novo ponto.
                        novoPesoMi = self._meuMetodoTC.calcularPesoMi(novoPrimalXi, novoFolgaDualSi)

                        vetorXYSCorrente = novoPonto

                        if(novoPesoMi > self._valorDeParada):
                            self._meuMetodoTC.atualizarMatrizJacobiana(jacobiana, novoPrimalXi, novoFolgaDualSi, dimA)
                        else:
                            print("Otimo Encontrado.")
                            procuraPontoMelhor = False
                    else:
                        print("Nao hah como melhorar a funcao objetivo andando nesta direcao.")
                        procuraPontoMelhor = False
                else:
                    if( (passoXi <= 0) and (passoSi > 0) ):
                        print("Primal Inviavel e Dual Ilimitado.")

                    if ((passoXi > 0) and (passoSi <= 0)):
                        print("Dual Inviavel e Primal Ilimitado.")

                    procuraPontoMelhor = False
            else:
                print("Jacobiana LD.")
                procuraPontoMelhor = False

            # Incrementando quantidade de iteracoes.
            iteracao = iteracao + 1
            print()

        return vetorXYSCorrente

if(__name__ == "__main__"):
    nomeArquivo = "Instancias\\instancia8.txt"
    m, n, c, b, A = lerInstanciaMPIFormato1(nomeArquivo)
    tal = 0.4
    valorDeParada = 1E-4
    epsZero = 1E-6

    vetorXYS = np.ones((2*n + m, 1))

    meuTCInviavel = MetodoTCInviavel(A, b, c, vetorXYS, tal, valorDeParada)

    (comecoEfimX, comecoEfimY, comecoEfimS) = meuTCInviavel._meuMetodoTC.localizarVariaveisXYS(np.shape(A))

    # Marcando inicio da contagem de tempo.
    tempoInicio = time.process_time()

    vetorXYSSolucao = meuTCInviavel.aplicarMetodoTCInviavel(epsZero)

    # Marcando fim da contagem do fim.
    tempoFim = time.process_time()

    tempoTotal = tempoFim - tempoInicio

    jacobiana = meuTCInviavel._meuMetodoTC.montarMinhaJacobiana()

    xOtimo = vetorXYSSolucao[comecoEfimX[0]:comecoEfimX[1] + 1]
    yOtimo = vetorXYSSolucao[comecoEfimY[0]:comecoEfimY[1] + 1]

    viabilidadeXotimo = np.linalg.norm((np.dot(A, xOtimo) - b))
    print("Viabilidade xOtimo")
    print(viabilidadeXotimo)

    print("Valor Otimo")
    print(float(np.dot(c.transpose(), xOtimo)))

    print("TempoTotal")
    print(tempoTotal)

    gap = meuTCInviavel._meuMetodoTC.calcularGapPrimalDual(xOtimo, yOtimo, b, c)
    print("gap")
    print(gap)

    # Ideia para verificar se o método encontrou o valor ótimo:
    #   -> Temos os valores das variaveis primais 'xi' e os valores das variaveis duais 'yi'. Portanto, podemos verificar
    # se o gap entre o valor da solucao primal e o valor da solucao dual é 0 (zero) ou bem proximo de 0 (zero). Neste caso,
    # como os problemas usados são de maximização, fazemos:
    # gap = b_transposto * yi - c_transposto * xi
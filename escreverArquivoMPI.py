import numpy as np
from geradorInstanciasMP import gerarInstanciaUniforme
from geradorInstanciasMP import gerarInstanciaNormal
from geradorInstanciasMP import geradorJefferson
from geradorInstanciasMP import gerarInstanciasViavel

def escreverInstanciaMPI(nomeArquivo, matrizA, vetorBzinho, vetorCzinho):
    with open(nomeArquivo, 'w') as arquivoEscrita:
        dimA = np.shape(matrizA)
        numLinhas = int(dimA[0])
        numColunas = int(dimA[1])

        arquivoEscrita.write(str(numLinhas))
        arquivoEscrita.write("\n")

        # Separador de Objetos: uma quebra de linha
        arquivoEscrita.write("\n")

        arquivoEscrita.write(str(numColunas))
        arquivoEscrita.write("\n")

        # Separador de Objetos: uma quebra de linha
        arquivoEscrita.write("\n")

        stringCt = []
        # Lista dos valores do vetor 'c' como strings.
        for elemento in vetorCzinho:
            num = float(elemento)
            stringCt.append(str(num))
        # Transformando a lista de 'c' como strings numa string só.
        stringCt = " ".join(stringCt)

        arquivoEscrita.write(stringCt)
        arquivoEscrita.write("\n")

        # Separador de Objetos: uma quebra de linha
        arquivoEscrita.write("\n")

        stringBt = []
        # Lista dos valores do vetor 'b' como strings.
        for elemento in vetorBzinho:
            num = float(elemento)
            stringBt.append(str(num))
        # Transformando a lista de 'b' como strings numa string só.
        stringBt = " ".join(stringBt)

        arquivoEscrita.write(stringBt)
        arquivoEscrita.write("\n")

        # Separador de Objetos: uma quebra de linha
        arquivoEscrita.write("\n")

        # Lista dos valores da matriz 'A' como strings.
        for linhaA in matrizA:
            stringA = []
            for elemento in linhaA:
                num = float(elemento)
                stringA.append(str(num))

            # Transformando a lista da linha corrente de 'A' como strings numa string só.
            stringA = " ".join(stringA)
            arquivoEscrita.write(stringA)
            arquivoEscrita.write("\n")
    arquivoEscrita.closed

if(__name__ == "__main__"):
    nomeArquivo = "Instancias\\instanciaTeste.txt"
    numLinhas = 40
    numColunas = 120

    matrizA, vetorBzinho, vetorCzinho = gerarInstanciasViavel(numLinhas, numColunas)

    escreverInstanciaMPI(nomeArquivo, matrizA, vetorBzinho, vetorCzinho)
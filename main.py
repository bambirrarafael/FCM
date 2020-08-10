import numpy as np
import pandas as pd
# import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import MinMaxScaler


def criar_normalizador(definicao_conceitos,  range=(0, 1)):
    """
    :param dataset: dataframe - cada coluna um conceito, primeira linha valor mínimo, segunda com o máximo
    :param range:
    :return:
    """
    normalizador = MinMaxScaler(feature_range=range)
    normalizador.fit(definicao_conceitos)
    return normalizador


def normalizar_linha(normalizador, linha):
    x = np.array(linha)
    x = x.reshape(1, -1)
    a = normalizador.transform(x)
    return a


def sigmoid(x, k=3):
    return 1/(1 + np.exp(-k * x))


def funcao_min(x, c):
    return np.sum(x * c)

def funcao_max(x, c):
    return np.sum(x * -c)

def maxmin(x, c, n_fo, mono_obj_sol):
    mu = np.zeros(n_fo)
    mu[0] = (mono_obj_sol[0, 0] - funcao_min(x, c[0:3])) / (mono_obj_sol[0, 0] - mono_obj_sol[0, 1])            # min
    mu[1] = (mono_obj_sol[1, 0] - funcao_min(x, c[3:6])) / (mono_obj_sol[1, 0] - mono_obj_sol[1, 1])            # min
    d = np.min(mu)
    return -d


def constraint(x):
    return np.sum(x) - 30


def simular_cenario_base(horizonte, matriz_de_pesos, definicao_conceitos, normalizador):
    cenario_normalizado = np.zeros((horizonte, len(matriz_de_pesos)))
    k = 3
    nomes = definicao_conceitos.columns
    a = definicao_conceitos.iloc[-1]
    cenario_normalizado[0, :] = normalizar_linha(normalizador, a)
    for i in range(1, len(cenario_normalizado)):
        #
        # ==============================================================================================================
        # encontrar a solucão harmoniosa de f1 e f2 para x1, x2 e x3
        a = [funcao_min, funcao_min]
        a_inv = [funcao_max, funcao_max]
        goal = [funcao_min, funcao_min]
        mono_obj_sol = np.zeros([len(goal), len(goal)])
        x0 = np.zeros(3)
        cons = [{'type': 'eq', 'fun': constraint}]
        bounds = [(0, 10), (0, 12), (0, 14)]
        opt_meth = "SLSQP"
        linha_anterior_desnormalizada = normalizador.inverse_transform([cenario_normalizado[i - 1]])
        # encontrar maximo
        mono_obj_sol[0, 0] = opt.minimize(a_inv[0], x0=x0, args=(linha_anterior_desnormalizada[0][0:3]),
                                          constraints=cons,
                                          bounds=bounds,
                                          method=opt_meth).fun
        mono_obj_sol[1, 0] = opt.minimize(a_inv[1], x0=x0, args=(linha_anterior_desnormalizada[0][3:6]),
                                          constraints=cons,
                                          bounds=bounds,
                                          method=opt_meth).fun
        mono_obj_sol[0, 0] = -mono_obj_sol[0, 0]
        mono_obj_sol[1, 0] = -mono_obj_sol[1, 0]
        # encontrar minimo
        mono_obj_sol[0, 1] = opt.minimize(a[0], x0=x0, args=(linha_anterior_desnormalizada[0][0:3]),
                                          constraints=cons,
                                          bounds=bounds,
                                          method=opt_meth).fun
        mono_obj_sol[1, 1] = opt.minimize(a[1], x0=x0, args=(linha_anterior_desnormalizada[0][3:6]),
                                          constraints=cons,
                                          bounds=bounds,
                                          method=opt_meth).fun
        # encontrar variaveis
        x0inp = opt.minimize(a[0], x0=x0, args=(linha_anterior_desnormalizada[0][0:3]),
                                          constraints=cons,
                                          bounds=bounds,
                                          method=opt_meth).x
        harm_sol = opt.minimize(maxmin, x0=x0inp, args=(linha_anterior_desnormalizada[0], 2, mono_obj_sol),
                                      constraints=cons,
                                      bounds=bounds, method=opt_meth).x
        # adicionalas ao cenário
        for var in range(3):
            harm_sol[var] = (harm_sol[var] - bounds[var][0])/(bounds[var][1] - bounds[var][0])    # normalizar de volta para [0, 1] para continuar a evolução do mapa
        # todo fix function values over time
        cenario_normalizado[i - 1][6:9] = harm_sol      # ajustes de variavel no mapa ok!
        # ==============================================================================================================
        for j in range(len(matriz_de_pesos)):
            x = np.matmul(cenario_normalizado[i-1, :], matriz_de_pesos[:, j])
            cenario_normalizado[i, j] = sigmoid(x, k)
    cenario_desnormalizado = normalizador.inverse_transform(cenario_normalizado)
    resultado = pd.DataFrame(cenario_desnormalizado, columns=nomes)
    #
    # ==================================================================================================================

    return resultado


def simular_outros_cenario(horizonte,
                           matriz_de_pesos,
                           definicao_conceitos,
                           normalizador,
                           normalizador_do_cenario_de_avaliacao,
                           cenario_de_analise
                           ):
    cenario_normalizado = np.zeros((horizonte, len(matriz_de_pesos)))
    lambd = 3
    nomes = definicao_conceitos.columns
    a = definicao_conceitos.iloc[-1]
    cenario_de_analise_normalizado = normalizador_do_cenario_de_avaliacao.transform(cenario_de_analise)
    cenario_de_analise_normalizado = pd.DataFrame(cenario_de_analise_normalizado, columns=cenario_de_analise.columns)
    cenario_normalizado[0, :] = normalizar_linha(normalizador, a)
    for i in range(1, len(cenario_normalizado)):
        for j in range(len(matriz_de_pesos)):
            x = np.matmul(cenario_normalizado[i-1, :], matriz_de_pesos[:, j])
            cenario_normalizado[i, j] = sigmoid(x, lambd)
            for k in cenario_de_analise:
                if nomes[j] == k:
                    cenario_normalizado[i, j] = cenario_de_analise_normalizado[k][i]
    cenario_desnormalizado = normalizador.inverse_transform(cenario_normalizado)
    resultado = pd.DataFrame(cenario_desnormalizado, columns=nomes)
    return resultado

def criar_matriz_de_pesos_aleatorios(inf, sup, n):
    np.random.seed(1)
    matriz = np.random.randint(inf, sup, [n, n]) / 100
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            if i == j:
                matriz[i, j] = 0
    return matriz

#
# criar matriz de pesos aleatórios e ajustar as variaveis e funções
matriz_de_pesos = criar_matriz_de_pesos_aleatorios(0, 100, 11)
# variaveis
matriz_de_pesos[6, :] = 0
matriz_de_pesos[7, :] = 0
matriz_de_pesos[8, :] = 0
matriz_de_pesos[:, 6] = 0
matriz_de_pesos[:, 7] = 0
matriz_de_pesos[:, 8] = 0
# FOs
matriz_de_pesos[9, :] = 0
matriz_de_pesos[10, :] = 0
matriz_de_pesos[:, 9] = 0
matriz_de_pesos[:, 10] = 0
matriz_de_pesos[0:3, 9] = 1
matriz_de_pesos[6:9, 9] = 1
matriz_de_pesos[3:9, 10] = 1

#
definicao_conceitos = pd.read_excel("./data/definicao_dos_conceitos.xlsx", index_col=0)
cenario_teste = pd.read_excel("./data/cenario_teste.xlsx")
horizonte = 30
#
# criar normalizador
normalizador = criar_normalizador(definicao_conceitos)
conceitos_para_cenario = definicao_conceitos[cenario_teste.columns]
normalizador_do_cenario_de_avaliacao = criar_normalizador(conceitos_para_cenario)

cenario_base = simular_cenario_base(
    horizonte,
    matriz_de_pesos,
    definicao_conceitos,
    normalizador
)

cenario_demanda_crescente = simular_outros_cenario(
    horizonte,
    matriz_de_pesos,
    definicao_conceitos,
    normalizador,
    normalizador_do_cenario_de_avaliacao,
    cenario_teste
)


# TODO
# alterar relações de velocidade do vento
# el ninõ e la ninã
# separar aneel
# estação do anoe relação com velocidade de vento

# escrever função de risco em função dos cenários




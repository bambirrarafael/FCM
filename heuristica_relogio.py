import copy as cp
import numpy as np


def muD_max(F, x):
    x = np.array(x)
    return np.min([f_.mu(x) for f_ in F])


def muD_min(F, x):
    x = np.array(x)
    return np.max([f_.mu(x) * -1 for f_ in F])


def heuristica_relogio(F, posicao_inicial, limites, passo=0.5):

    def possivel_subtrair(posicao, limite, passo):
        return True if posicao - passo >= limite[0] else False

    def possivel_adicionar(posicao, limite, passo):
        return True if posicao + passo <= limite[1] else False

    def caminha(posicao_origem, passo, subtrair_de, adicionar_em):
        posicao_destino = cp.deepcopy(posicao_origem)
        posicao_destino[subtrair_de] -= passo
        posicao_destino[adicionar_em] += passo
        return posicao_destino

    def passo_e_melhor(F, posicao_origem, posicao_destino):
        return (
            True if muD_max(F, posicao_destino) > muD_max(F, posicao_origem)
            else False
        )

    posicao_atual = np.array(cp.deepcopy(posicao_inicial))
    num_vars = len(posicao_atual)
    i = 0

    while i < num_vars:
        j = 0
        while j < num_vars:
            if i != j:
                if possivel_subtrair(posicao_atual[i], limites[i], passo):
                    if possivel_adicionar(posicao_atual[j], limites[j], passo):
                        if passo_e_melhor(F,
                                          posicao_atual,
                                          caminha(posicao_atual, passo, i, j)):
                            posicao_atual = caminha(posicao_atual, passo, i, j)
                        else:
                            j += 1
                    else:
                        j += 1
                else:
                    i += 1
                    break
            else:
                j += 1
        i += 1
    return posicao_atual
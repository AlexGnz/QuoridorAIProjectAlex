'''

Projet d'année partie 4
Alexandre Gonze - 439738
BA1 INFO
07/04/2019

'''


import numpy as np
import random

EPS_MIN = 0.06  # the lowest the epsilon-greedy parameter can go

'''
NOUVELLE METHODE
Possibilité de changer de type de fonction d'activation
'''
def activationFunction(x, type = None):
    if type is not None:
        if type == 'TANH':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif type == 'SWISH':
            return (x / (1 + np.exp(-x)))
    return 1 / (1 + np.exp(-x))


'''
NOUVELLE METHODE
Il fallait également adapter la dérivée des fonctions d'activation en fonction du type sélectionné!
'''
def derivatedActivationFunction(x, type = None):
    if type is not None:
        if type == 'TANH':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif type == 'SWISH':
            return (np.exp(x)*(np.exp(x)+x+1))/((np.exp(x)+1)**2)
    return x * (1 - x)

def initWeights(nb_rows, nb_columns):
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    W_int = initWeights(n_hidden, n_input)
    W_out = initWeights(n_hidden, 1)[:, 0]
    return (W_int, W_out)


def forwardPass(s, NN, type = None):
    W_int = NN[0]
    W_out = NN[1]
    P_int = activationFunction(np.dot(W_int, s), type)
    p_out = activationFunction(P_int.dot(W_out), type)
    return p_out


def backpropagation(s, NN, delta, learning_strategy=None, type = None):
    if learning_strategy is None:
        return None
    W_int = NN[0]
    W_out = NN[1]
    P_int = activationFunction(np.dot(W_int, s), type)
    p_out = activationFunction(P_int.dot(W_out), type)
    grad_out = derivatedActivationFunction(p_out, type)
    grad_int = derivatedActivationFunction(P_int, type)
    Delta_int = grad_out * W_out * grad_int
    if learning_strategy[0] == 'Q-learning':
        alpha = learning_strategy[1]
        W_int -= alpha * delta * np.outer(Delta_int, s)
        W_out -= alpha * delta * grad_out * P_int
    elif learning_strategy[0] == 'TD-lambda' or learning_strategy[0] == 'Q-lambda':
        alpha = learning_strategy[1]
        lamb = learning_strategy[2]
        Z_int = learning_strategy[3]
        Z_out = learning_strategy[4]
        Z_int *= lamb
        Z_int += np.outer(Delta_int, s)
        Z_out *= lamb
        Z_out += grad_out * P_int
        W_int -= alpha * delta * Z_int
        W_out -= alpha * delta * Z_out


'''
METHODE MODIFIEE
- Ajout d'une variante de eps-greedy : On prend le maximum entre un eps minimum définit et 0.99 exposant le nombre de parties déjà effectuées, multipliées par eps 
Cela réduit eps petit à petit, jusqu'à arriver au eps minimum
- Ajout de Q-lambda, qui remet les eligibility traces à 0 quand on choisit un 
'''
def makeMove(moves, s, color, NN, eps, learning_strategy=None, numberTrains=0, type = None):
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    Q_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'Q-lambda')
    # Epsilon greedy décroissant : Permet de fort inciter à l'exploration au début et réduit au fur et à mesure la probabilité d'effectuer des coups non optimaux.
    # Affine donc ses mouvements au fil du temps
    # En ayant un epsilon constant, il faut trouver le juste milieu entre exploitation et exploration, ici on fait les deux, au fur et à mesure
    eps = max((0.99 ** numberTrains) * eps, EPS_MIN)
    greedy = random.random() > eps

    if greedy or Q_learning:
        best_moves = []
        best_value = None
        c = 1
        if color == 1:
            c = -1
        for m in moves:
            val = forwardPass(m, NN, type)
            if best_value == None or c * val > c * best_value:
                best_moves = [m]
                best_value = val
            elif val == best_value:
                best_moves.append(m)
    if greedy:
        new_s = best_moves[random.randint(0, len(best_moves) - 1)]
    else:
        new_s = moves[random.randint(0, len(moves) - 1)]
    if Q_learning or TD_lambda or Q_lambda:
        p_out_s = forwardPass(s, NN, type)
        if Q_learning:
            delta = p_out_s - best_value
        elif TD_lambda or Q_lambda:
            if greedy:
                p_out_new_s = best_value
            else:
                p_out_new_s = forwardPass(new_s, NN, type)
                if Q_lambda:
                    # Attention, seulement si ce n'est pas le meilleur mouvement qui aurait pu etre pris aléatoirement
                    best_value_2 = getBestValue(moves, NN, color, type)
                    if p_out_new_s < best_value_2:
                        eligibilityTracesToZero(learning_strategy)
            delta = p_out_s - p_out_new_s
        backpropagation(s, NN, delta, learning_strategy, type)

    return new_s

'''
NOUVELLE METHODE
Permet de déterminer si le mouvement aléatoire choisi était le mouvement optimal, ou pas, et d'ainsi mettre les eligibility traces à 0, ou pas
'''
def getBestValue(moves, NN, color, type = None):
    best_value = None
    c = 1
    if color == 1:
        c = -1
    for m in moves:
        val = forwardPass(m, NN, type)
        if best_value == None or c * val > c * best_value:
            best_value = val
    return best_value

'''
METHODE MODIFIEE
Ajout de la stratégie d'apprentissage Q-lambda
'''
def endGame(s, won, NN, learning_strategy, type = None):
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    Q_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'Q-lambda')
    if Q_learning or TD_lambda or Q_lambda:
        p_out_s = forwardPass(s, NN, type)
        delta = p_out_s - won
        backpropagation(s, NN, delta, learning_strategy, type)
        if TD_lambda or Q_lambda:
            eligibilityTracesToZero(learning_strategy)

'''
NOUVELLE METHODE
Simplement pour remettre les eligibility traces à zéro, vu que c'est utilisé à plusieurs endroits, on en fait une fonction
'''
def eligibilityTracesToZero(learning_strategy):
    learning_strategy[3].fill(0)
    learning_strategy[4].fill(0)

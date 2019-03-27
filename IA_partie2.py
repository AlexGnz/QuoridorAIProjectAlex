import numpy as np
import random

EPS_MIN = 0.06    # the lowest the epsilon-greedy parameter can go

def sigmoid(x):
    #return (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
    return 1 / (1 + np.exp(-x))

def initWeights(nb_rows,nb_columns):
    return np.random.normal(0, 0.0001, (nb_rows,nb_columns))    

def createNN(n_input, n_hidden):
    W_int = initWeights(n_hidden,n_input)  
    W_out = initWeights(n_hidden,1)[:,0]
    return (W_int, W_out)

def forwardPass(s, NN):
    W_int = NN[0]
    W_out = NN[1]
    P_int = sigmoid(np.dot(W_int,s))
    p_out = sigmoid(P_int.dot(W_out))
    return p_out

def backpropagation(s, NN, delta, learning_strategy=None):
    if learning_strategy is None:
        return None
    W_int = NN[0]
    W_out = NN[1]                   
    P_int = sigmoid(np.dot(W_int,s))                
    p_out = sigmoid(P_int.dot(W_out))
    #Avec tangente hyperbolique
    grad_out = 1 - p_out**2
    grad_int = 1 - P_int**2
    #Avec sigmoide
    grad_out = p_out*(1-p_out)
    grad_int = P_int*(1-P_int)
    Delta_int = grad_out*W_out*grad_int   
    if learning_strategy[0] == 'Q-learning':
        alpha = learning_strategy[1]
        W_int -= alpha*delta*np.outer(Delta_int,s)            
        W_out -= alpha*delta*grad_out*P_int 
    elif learning_strategy[0] == 'TD-lambda':              
        alpha = learning_strategy[1]
        lamb = learning_strategy[2]
        Z_int = learning_strategy[3]
        Z_out = learning_strategy[4]    
        Z_int *= lamb
        Z_int += np.outer(Delta_int,s)
        Z_out *= lamb
        Z_out += grad_out*P_int
        W_int -= alpha*delta*Z_int
        W_out -= alpha*delta*Z_out


def makeMove(moves, s, color, NN, eps, learning_strategy=None, numberTrains = 0):
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    # Epsilon greedy
    # Quand on compare 2 IA, on n'utilise plus cette formule
    if eps != 0.05:
        eps = max((0.9999**numberTrains) * eps, EPS_MIN)
    greedy = random.random() > eps
    #print(eps)

    # dans le cas greedy, on recherche le meilleur mouvement (état) possible. Dans le cas du Q-learning (même sans greedy), on a besoin de connaître
    # la probabilité estimée associée au meilleur mouvement (état) possible en vue de réaliser la backpropagation.
    if greedy or Q_learning:
        best_moves = []
        best_value = None
        c = 1
        if color == 1:
            # au cas où c'est noir qui joue, on s'interessera aux pires coups du point de vue de blanc
            c = -1
        for m in moves:
            val = forwardPass(m, NN)
            if best_value == None or c*val > c*best_value: #si noir joue, c'est comme si on regarde alors si val < best_value
                best_moves = [ m ]
                best_value = val
            elif val == best_value: 
                best_moves.append(m)                                      
    if greedy:
        # on prend un mouvement au hasard parmi les meilleurs (pires si noir)
        new_s = best_moves[ random.randint(0, len(best_moves)-1) ]            
    else:
        # on choisit un mouvement au hasard
        new_s = moves[ random.randint(0, len(moves)-1) ]                
    # on met à jour les poids si nécessaire
    if Q_learning or TD_lambda:        
        p_out_s = forwardPass(s, NN)
        if Q_learning:                    
            delta = p_out_s - best_value            
        elif TD_lambda:                      
            if greedy:
                p_out_new_s = best_value
            else:
                p_out_new_s = forwardPass(new_s, NN)
            delta = p_out_s - p_out_new_s            
        backpropagation(s, NN, delta, learning_strategy)        

    return new_s        

def endGame(s, won, NN, learning_strategy):
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    # on met à jour les poids si nécessaire
    if Q_learning or TD_lambda:
        p_out_s = forwardPass(s, NN)
        delta = p_out_s - won
        backpropagation(s, NN, delta, learning_strategy)            
        if TD_lambda:                      
            # on remet les eligibility traces à 0 en prévision de la partie suivante            
            learning_strategy[3].fill(0) # remet Z_int à 0
            learning_strategy[4].fill(0) # remet Z_out à 0
            


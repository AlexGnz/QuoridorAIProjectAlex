'''

Projet d'année partie 4
Alexandre Gonze - 439738
BA1 INFO
07/04/2019

'''


import os, sys, time, random
import numpy as np
'''
CONFIG TOURNOI
'''
N = 5  # N*N board
WALLS = 3  # number of walls each player has
EPS = 0.5 # Quand on utilise eps-greedy décroissant
EPS_MIN = 0.06  # the lowest the epsilon-greedy parameter can go
#EPS = 0.2  # Quand on utilise epsilon constant epsilon in epsilon-greedy
ALPHA = 0.4  # learning_rate
LAMB = 0.9  # lambda for TD(lambda)
LS = 'Q-lambda'  # default learning strategy
G = None  # graph of board (used for connectivity checks)
G_INIT = None  # graph of starting board
TYPE = 'SIGMOID'


'''
Vu que je ne savais pas si le fichier tournoi.py était pris tout seul ou runné dans un dossier avec IA_partie4.py, j'ai préféré mettre toutes les fonctions ici
J'ai également retiré les commentaires de toutes les fonctions présentes ici. Cfr IA_partie4.py 
'''

def activationFunction(x, type = None):
    if type is not None:
        if type == 'TANH':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif type == 'SWISH':
            return (x / (1 + np.exp(-x)))
    return 1 / (1 + np.exp(-x))

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
                    best_value_2 = getBestValue(moves, NN, color, type)
                    if p_out_new_s < best_value_2:
                        eligibilityTracesToZero(learning_strategy)
            delta = p_out_s - p_out_new_s
        backpropagation(s, NN, delta, learning_strategy, type)

    return new_s

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

def eligibilityTracesToZero(learning_strategy):
    learning_strategy[3].fill(0)
    learning_strategy[4].fill(0)


def clearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')

def progressBar(i, n):
    if int(100 * i / n) > int(100 * (i - 1) / n):
        print('  ' + str(int(100 * i / n)) + '%', end='\r')

class Player_AI():
    def __init__(self, NN, eps, learning_strategy, name='IA'):
        self.name = name
        self.color = None  # white (0) or black(1)
        self.score = 0
        self.NN = NN
        self.eps = eps
        self.learning_strategy = learning_strategy
        self.numberTrains = 0

    def makeMove(self, board):
        return makeMove(listMoves(board, self.color), board, self.color, self.NN, self.eps, self.learning_strategy,
                        self.numberTrains, TYPE)

    def endGame(self, board, won):
        # print(self.numberTrains)
        self.numberTrains += 1
        endGame(board, won, self.NN, self.learning_strategy, TYPE)


def listEncoding(board):
    pos = [None, None]
    coord = [[None, None], [None, None]]
    walls = [[], []]
    walls_left = [None, None]
    for i in range(2):
        pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
        coord[i][0] = pos[i] % N
        coord[i][1] = pos[i] // N
        for j in range((N - 1) ** 2):
            if board[2 * N ** 2 + i * (N - 1) ** 2 + j] == 1:
                walls[i].append([j % (N - 1), j // (N - 1)])
        walls_left[i] = board[
                        2 * N ** 2 + 2 * (N - 1) ** 2 + i * (WALLS + 1):2 * N ** 2 + 2 * (N - 1) ** 2 + (i + 1) * (
                                    WALLS + 1)].argmax()
    return [coord[0], coord[1], walls[0], walls[1], walls_left[0], walls_left[1]]



def eachPlayerHasPath(board):
    global N, WALLS, G
    nb_walls = board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2].sum()
    if nb_walls <= 1:
        return True
    pos = [None, None]
    coord = [[None, None], [None, None]]
    for i in range(2):
        pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
        coord[i][0] = pos[i] % N
        coord[i][1] = pos[i] // N
        coord[i] = np.array(coord[i])
    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i in range(len(steps)):
        steps[i] = np.array(steps[i])
    for i in range(2):
        A = np.zeros((N, N), dtype='bool')
        S = [coord[i]]
        finished = False
        while len(S) > 0 and not finished:
            c = S.pop()
            A[c[1]][c[0]] = True
            for k in range(4):
                if G[c[0]][c[1]][k] == 1:
                    s = steps[k]
                    new_c = c + s
                    if i == 0:
                        if new_c[1] == N - 1:
                            finished = True
                            break
                    else:
                        if new_c[1] == 0:
                            return True
                    if A[new_c[1]][new_c[0]] == False:
                        if i == 0:
                            if k == 1:
                                S.append(new_c)
                            else:
                                S.insert(0, new_c)
                        else:
                            if k == 3:
                                S.append(new_c)
                            else:
                                S.insert(0, new_c)
        if not finished:
            return False
    return True


def canMove(board, coord, step):
    new_coord = coord + step
    in_board = new_coord.min() >= 0 and new_coord.max() <= N - 1
    if not in_board:
        return False
    if WALLS > 0:
        if step[0] == -1:
            L = []
            if new_coord[1] < N - 1:
                L.append(2 * N ** 2 + (N - 1) ** 2 + new_coord[1] * (N - 1) + new_coord[0])
            if new_coord[1] > 0:
                L.append(2 * N ** 2 + (N - 1) ** 2 + (new_coord[1] - 1) * (N - 1) + new_coord[0])
        elif step[0] == 1:
            L = []
            if coord[1] < N - 1:
                L.append(2 * N ** 2 + (N - 1) ** 2 + coord[1] * (N - 1) + coord[0])
            if coord[1] > 0:
                L.append(2 * N ** 2 + (N - 1) ** 2 + (coord[1] - 1) * (N - 1) + coord[0])
        elif step[1] == -1:
            L = []
            if new_coord[0] < N - 1:
                L.append(2 * N ** 2 + new_coord[1] * (N - 1) + new_coord[0])
            if new_coord[0] > 0:
                L.append(2 * N ** 2 + new_coord[1] * (N - 1) + new_coord[0] - 1)
        elif step[1] == 1:
            L = []
            if coord[0] < N - 1:
                L.append(2 * N ** 2 + coord[1] * (N - 1) + coord[0])
            if coord[0] > 0:
                L.append(2 * N ** 2 + coord[1] * (N - 1) + coord[0] - 1)
        else:
            print('step vector', step, 'is not valid')
            quit(1)
        if sum([board[j] for j in L]) > 0:
            return False
    return True


def computeGraph(board=None):
    global N, WALLS
    pos_steps = [(1, 0), (0, 1)]
    for i in range(len(pos_steps)):
        pos_steps[i] = np.array(pos_steps[i])
    g = np.zeros((N, N, 4))
    for i in range(N):
        for j in range(N):
            c = np.array([i, j])
            for k in range(2):
                s = pos_steps[k]
                if board is None:
                    new_c = c + s
                    if new_c.min() >= 0 and new_c.max() <= N - 1:
                        g[i][j][k] = 1
                        g[new_c[0]][new_c[1]][k + 2] = 1
                else:
                    if canMove(board, c, s):
                        new_c = c + s
                        g[i][j][k] = 1
                        g[new_c[0]][new_c[1]][k + 2] = 1
    return g


def listMoves(board, current_player):
    if current_player not in [0, 1]:
        print('error in function listMoves: current_player =', current_player)
    pn = current_player
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(len(steps)):
        steps[i] = np.array(steps[i])
    moves = []
    pos = [None, None]
    coord = [None, None]
    for i in range(2):
        pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
        coord[i] = np.array([pos[i] % N, pos[i] // N])
        pos[i] += pn * N ** 2
    P = []
    for s in steps:
        if canMove(board, coord[pn], s):
            new_coord = coord[pn] + s
            new_pos = pos[pn] + s[0] + N * s[1]
            occupied = np.array_equal(new_coord, coord[(pn + 1) % 2])
            if not occupied:
                P.append([pos[pn], new_pos])
            else:
                can_jump_straight = canMove(board, new_coord, s)
                if can_jump_straight:
                    new_pos = new_pos + s[0] + N * s[1]
                    P.append([pos[pn], new_pos])
                else:
                    if s[0] == 0:
                        D = [(-1, 0), (1, 0)]
                    else:
                        D = [(0, -1), (0, 1)]
                    for i in range(len(D)):
                        D[i] = np.array(D[i])
                    for d in D:
                        if canMove(board, new_coord, d):
                            final_pos = new_pos + d[0] + N * d[1]
                            P.append([pos[pn], final_pos])
    nb_walls_left = board[2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1):2 * N ** 2 + 2 * (N - 1) ** 2 + (pn + 1) * (
                WALLS + 1)].argmax()
    ind_walls_left = 2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1) + nb_walls_left
    if nb_walls_left > 0:
        for i in range(2 * (N - 1) ** 2):
            pos = 2 * N ** 2 + i
            L = [pos]
            if i < (N - 1) ** 2:
                L.append(pos + (N - 1) ** 2)
                if i % (N - 1) > 0:
                    L.append(pos - 1)
                if i % (N - 1) < N - 2:
                    L.append(pos + 1)
            else:
                L.append(pos - (N - 1) ** 2)
                if (i - (N - 1) ** 2) // (N - 1) > 0:
                    L.append(pos - (N - 1))
                if (i - (N - 1) ** 2) // (N - 1) < N - 2:
                    L.append(pos + (N - 1))
            nb_intersecting_wall = sum([board[j] for j in L])
            if nb_intersecting_wall == 0:
                board[pos] = 1
                if i < (N - 1) ** 2:
                    a, b = i % (N - 1), i // (N - 1)
                    E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                else:
                    a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                    E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                for e in E:
                    G[e[0]][e[1]][e[2]] = 0
                if eachPlayerHasPath(board):
                    P.append([pos, ind_walls_left - 1, ind_walls_left])  # put down the wall and adapt player's counter
                board[pos] = 0
                for e in E:
                    G[e[0]][e[1]][e[2]] = 1
    for L in P:
        new_board = board.copy()
        for i in L:
            new_board[i] = not new_board[i]
        moves.append(new_board)

    return moves


def endOfGame(board):
    return board[(N - 1) * N:N ** 2].max() == 1 or board[N ** 2:N ** 2 + N].max() == 1


def startingBoard():
    board = np.array([0] * (2 * N ** 2 + 2 * (N - 1) ** 2 + 2 * (WALLS + 1)))
    board[(N - 1) // 2] = True
    board[N ** 2 + N * (N - 1) + (N - 1) // 2] = True
    for i in range(2):
        board[2 * N ** 2 + 2 * (N - 1) ** 2 + i * (WALLS + 1) + WALLS] = 1
    return board


def playGame(player1, player2, show=False, delay=0.0):
    global N, WALLS, G, G_INIT
    players = [player1, player2]
    board = startingBoard()
    G = G_INIT.copy()
    for i in range(2):
        players[i].color = i
    finished = False
    current_player = 0
    count = 0
    quit = False
    while not finished:
        if show:
            msg = ''
            txt = ['Blanc', 'Noir ']
            for i in range(2):
                if i == current_player:
                    msg += '* '
                else:
                    msg += '  '
                msg += txt[i] + ' : ' + players[i].name
                msg += '\n'
            for i in range(2):
                if players[i].name == 'IA':
                    p = forwardPass(board, players[i].NN, TYPE)
                    msg += '\nEstimation IA : ' + "{0:.4f}".format(p)
                    msg += '\n'
            time.sleep(delay)
        new_board = players[current_player].makeMove(board)
        if not new_board is None:
            v = new_board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2] - board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2]
            i = v.argmax()
            if v[i] == 1:
                if i < (N - 1) ** 2:
                    a, b = i % (N - 1), i // (N - 1)
                    E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                else:
                    a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                    E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                for e in E:
                    G[e[0]][e[1]][e[2]] = 0
        board = new_board
        if board is None:
            quit = True
            finished = True
        elif endOfGame(board):
            players[current_player].score += 1
            white_won = current_player == 0
            players[current_player].endGame(board, white_won)
            if show:
                time.sleep(0.3)
            finished = True
        else:
            current_player = (current_player + 1) % 2
    return quit


def train(NN, n_train=10000):
    if LS == 'Q-learning':
        learning_strategy1 = (LS, ALPHA)
        learning_strategy2 = (LS, ALPHA)
    elif LS == 'TD-lambda' or LS == 'Q-lambda':
        learning_strategy1 = (LS, ALPHA, LAMB, np.zeros(NN[0].shape), np.zeros(NN[1].shape))
        learning_strategy2 = (LS, ALPHA, LAMB, np.zeros(NN[0].shape), np.zeros(NN[1].shape))
    agent1 = Player_AI(NN, EPS, learning_strategy1, 'agent 1')
    agent2 = Player_AI(NN, EPS, learning_strategy2, 'agent 2')
    print('\nEntraînement (' + str(n_train) + ' parties)')
    for j in range(n_train):
        progressBar(j, n_train)
        playGame(agent1, agent2)

def main():
    global G_INIT
    NN = createNN(2 * 5 ** 2 + 2 * (5 - 1) ** 2 + 2 * (3 + 1), 40)

    G_INIT = computeGraph()

    train(NN, int(sys.argv[1]))

    np.savez(sys.argv[2], N=5, WALLS=3, W1=NN[0], W2=NN[1])
    # Comme demandé, cela print le type de fonction d'activation demandée
    print('IA entrainée avec la fonction d\'activation suivante : ',TYPE)

if __name__ == '__main__':
    main()



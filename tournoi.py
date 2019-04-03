import copy, os, time, sys, random, time
from select import select
import numpy as np

EPS_MIN = 0.06

def sigmoid(x):
    # return (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
    return 1 / (1 + np.exp(-x))


def initWeights(nb_rows, nb_columns):
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    W_int = initWeights(n_hidden, n_input)
    W_out = initWeights(n_hidden, 1)[:, 0]
    return (W_int, W_out)


def forwardPass(s, NN):
    W_int = NN[0]
    W_out = NN[1]
    P_int = sigmoid(np.dot(W_int, s))
    p_out = sigmoid(P_int.dot(W_out))
    return p_out


def backpropagation(s, NN, delta, learning_strategy=None):
    if learning_strategy is None:
        return None
    W_int = NN[0]
    W_out = NN[1]
    P_int = sigmoid(np.dot(W_int, s))
    p_out = sigmoid(P_int.dot(W_out))
    # Avec tangente hyperbolique
    grad_out = 1 - p_out ** 2
    grad_int = 1 - P_int ** 2
    # Avec sigmoide
    grad_out = p_out * (1 - p_out)
    grad_int = P_int * (1 - P_int)
    Delta_int = grad_out * W_out * grad_int
    if learning_strategy[0] == 'Q-learning':
        alpha = learning_strategy[1]
        W_int -= alpha * delta * np.outer(Delta_int, s)
        W_out -= alpha * delta * grad_out * P_int
    elif learning_strategy[0] == 'TD-lambda':
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


def makeMove(moves, s, color, NN, eps, learning_strategy=None, numberTrains=0):
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    # Epsilon greedy
    # Quand on compare 2 IA, on n'utilise plus cette formule
    eps = max((0.99 ** numberTrains) * eps, EPS_MIN)
    greedy = random.random() > eps
    # print(eps)

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
            if best_value == None or c * val > c * best_value:  # si noir joue, c'est comme si on regarde alors si val < best_value
                best_moves = [m]
                best_value = val
            elif val == best_value:
                best_moves.append(m)
    if greedy:
        # on prend un mouvement au hasard parmi les meilleurs (pires si noir)
        new_s = best_moves[random.randint(0, len(best_moves) - 1)]
    else:
        # on choisit un mouvement au hasard
        new_s = moves[random.randint(0, len(moves) - 1)]
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
            learning_strategy[3].fill(0)  # remet Z_int à 0
            learning_strategy[4].fill(0)  # remet Z_out à 0


N = 5  # N*N board
WALLS = 3  # number of walls each player has
EPS = 0.2  # epsilon in epsilon-greedy
ALPHA = 0.4  # learning_rate
LAMB = 0.9  # lambda for TD(lambda)
LS = 'Q-learning'  # default learning strategy
G = None  # graph of board (used for connectivity checks)
G_INIT = None  # graph of starting board

# KEYS
UP = '\x1b[A'
DOWN = '\x1b[B'
LEFT = '\x1b[D'
RIGHT = '\x1b[C'
# NB: diagonal jumps are usually done using arrow keys by going to the opponent's position first, below: alternative keys
UP_LEFT = 'd'
UP_RIGHT = 'f'
DOWN_LEFT = 'c'
DOWN_RIGHT = 'v'
QUIT = 'q'


def clearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')


def waitForKey():
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save)  # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios.IGNCR
                  | termios.ICRNL | termios.IXON)
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios.PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1)  # returns a single character
        if ret == '\x1b':
            ret += sys.stdin.read(2)
    except KeyboardInterrupt:
        ret = '\x03'
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret


def wait(timeout):
    rlist, wlist, xlist = select([sys.stdin], [], [], timeout)


def progressBar(i, n):
    if int(100 * i / n) > int(100 * (i - 1) / n):
        print('  ' + str(int(100 * i / n)) + '%', end='\r')


class Player_Human():
    def __init__(self, name='Humain'):
        self.name = name
        self.color = None  # white (0) or black(1)
        self.score = 0

    def makeMove(self, board):
        global N, WALLS, G_INIT, G
        moves = listMoves(board, self.color)
        lmoves = [listEncoding(m) for m in moves]
        lboard = listEncoding(board)
        msg = 'Choisissez un mouvement (flèches pour bouger, w pour poser un mur, q pour quitter)'
        print(msg)
        D = [[LEFT, [-1, 0], None], [RIGHT, [1, 0], None], [UP, [0, 1], None], [DOWN, [0, -1], None],  # one step moves
             [LEFT, [-2, 0], None], [RIGHT, [2, 0], None], [UP, [0, 2], None], [DOWN, [0, -2], None],  # jumps
             [UP_LEFT, [-1, 1], None], [UP_RIGHT, [1, 1], None], [DOWN_LEFT, [-1, -1], None],
             [DOWN_RIGHT, [1, -1], None]]  # diagonal moves
        for i in range(len(D)):
            for j in range(len(lmoves)):
                m = moves[j]
                lm = lmoves[j]
                if list(np.array(lm[self.color])) == list(np.array(lboard[self.color]) + np.array(D[i][1])):
                    D[i][2] = m
                    break
        wall_moves = [lm for lm in lmoves if lm[self.color] == lboard[self.color]]
        wall_coord = [[], []]
        wall_hv = [[], []]
        for lm in wall_moves:
            i = int(lm[2] == lboard[2])
            wall_hv[i].append(lm)
            for c in lm[2 + i]:
                if c not in lboard[2 + i]:
                    break
            wall_coord[i].append(c)
        quit = False
        while not quit:
            key = waitForKey()
            if key == QUIT:
                quit = True
                break
            # player changes position:
            for i in range(len(D)):
                if key == D[i][0]:
                    if not (D[i][2] is None):
                        return D[i][2]
                    elif (i <= 3) and (D[i + 4][2] is None):
                        p = np.array(lboard[self.color]) + np.array(D[i][1])
                        q = np.array(lboard[(self.color + 1) % 2])
                        if p.tolist() == q.tolist():
                            # we moved to the opponent's position but the jump is blocked, we check if some diagonal move is possible
                            s = np.array(D[i][1])
                            diagonal_jump_feasible = False
                            for j in range(8, 12):
                                if not (D[j][2] is None):
                                    r = np.array(D[j][1])
                                    if r[0] == s[0] or r[1] == s[1]:
                                        diagonal_jump_feasible = True
                            if diagonal_jump_feasible:
                                halfway = board.copy()
                                halfway[self.color * N ** 2:self.color * N ** 2 + N ** 2] = halfway[((
                                                                                                                 self.color + 1) % 2) * N ** 2:(
                                                                                                                                                           (
                                                                                                                                                                       self.color + 1) % 2) * N ** 2 + N ** 2]
                                display(halfway, 'Saut diagonal, choisissez la destination')
                                second_key = waitForKey()
                                diagonal_jump = False
                                for j in range(4):
                                    if second_key == D[j][0]:
                                        t = np.array(D[j][1])
                                        r = s + t
                                        if abs(r[0]) == 1 and abs(r[1]) == 1:
                                            # diagonal jump selected, we check if that jump is feasible
                                            for k in range(8, 12):
                                                if r.tolist() == D[k][1] and not (D[k][2] is None):
                                                    key = D[k][0]
                                                    diagonal_jump = True
                                                    break
                                if not diagonal_jump:
                                    display(board, msg)

            # player puts down a wall
            if key == 'w' and lboard[4 + self.color] > 0 and len(wall_moves) > 0:
                msg = "Pose d'un mur (flèches pour faire defiler, w pour changer l'orientation, ENTER pour selectionner, a pour annuler, q pour quitter)"
                j = 0
                if len(wall_hv[0]) > 0:
                    h = 0
                else:
                    h = 1
                while not quit:
                    i = lmoves.index(wall_hv[h][j])
                    display(moves[i], msg)
                    key = waitForKey()
                    if key == QUIT:
                        quit = True
                        break
                    if key == 'a':
                        display(board)
                        break
                    elif key == 'w':
                        if len(wall_hv[(h + 1) % 2]) > 0:
                            c = wall_coord[h][j]
                            h = (h + 1) % 2
                            if c in wall_coord[h]:
                                j = wall_coord[h].index(c)
                            else:
                                best_d = wall_coord[h][0]
                                min_val = (abs(best_d[0] - c[0]) + abs(best_d[1] - c[1])) ** 2
                                for d in wall_coord[h]:
                                    val = (abs(d[0] - c[0]) + abs(d[1] - c[1])) ** 2
                                    if val < min_val:
                                        min_val = val
                                        best_d = d
                                j = wall_coord[h].index(best_d)
                    elif key == LEFT:
                        j = (j - 1) % len(wall_hv[h])
                    elif key == RIGHT:
                        j = (j + 1) % len(wall_hv[h])
                    elif key == UP:
                        c = wall_coord[h][j]
                        next_j = j
                        for k in range(j, len(wall_coord[h])):
                            if wall_coord[h][k][0] == c[0] and wall_coord[h][k][1] > c[1]:
                                next_j = k
                                break
                        j = next_j
                    elif key == DOWN:
                        c = wall_coord[h][j]
                        next_j = j
                        for k in range(j, -1, -1):
                            if wall_coord[h][k][0] == c[0] and wall_coord[h][k][1] < c[1]:
                                next_j = k
                                break
                        j = next_j
                    elif key == '\r':
                        return moves[i]
        if quit:
            return None

    def endGame(self, board, won):
        pass


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
                        self.numberTrains)

    def endGame(self, board, won):
        # print(self.numberTrains)
        self.numberTrains += 1
        endGame(board, won, self.NN, self.learning_strategy)


def listEncoding(board):
    # outputs list encoding of board:
    # [ [i, j], [k, l], list_of_horizontal_walls, list_of_vertical_walls, walls_left_p1, walls_left_p2 ]
    # where [i, j] position of white player and [k, l] position of black player
    # and each wall in lists of walls is of the form [a, b] where [a,b] is the south-west square
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


def display(board, msg=None):
    global N, WALLS
    lboard = listEncoding(board)
    clearScreen()
    square_char = '\u25A2'
    player_char = ('\u25CF', '\u25CD')  # empty circle: '\u25CB'
    wall_char = ('\u2014', '\uFF5C')
    h_offset = 8
    s = ''
    for j in range(N):
        line = ' ' * h_offset
        hwall = ' ' * h_offset
        for i in range(N):
            old_line = line
            for p in range(2):
                if [i, j] == lboard[p]:
                    line += player_char[p]
            if line == old_line:
                line += square_char
            beginning_vertical_wall = [i, j - 1] in lboard[3]
            ending_vertical_wall = j < N - 1 and [i, j] in lboard[3]
            if beginning_vertical_wall:
                line += ' ' + wall_char[1]
            elif ending_vertical_wall:
                line += ' ' + wall_char[1]
            else:
                line += '   '
            if j > 0:
                old_hwall = hwall
                for wall in lboard[2]:
                    if wall == [i, j - 1]:
                        hwall += wall_char[0] * 4
                    elif wall == [i - 1, j - 1]:
                        hwall += wall_char[0] * 2
                        if beginning_vertical_wall:
                            hwall += wall_char[1]
                        else:
                            hwall += ' ' * 2
                if hwall == old_hwall:
                    if beginning_vertical_wall:
                        hwall += '  ' + wall_char[1]
                    else:
                        hwall += ' ' * 4

        s = line + '\n' + hwall + '\n' + s
    H = [' ' * (h_offset - 2)] * 2
    for pn in range(2):
        H[pn] += (wall_char[1] + '  ') * lboard[4 + pn]
    s = (H[1] + '\n') * 2 + '\n' + s + (H[0] + '\n') * 2
    print('\n' * 2 + s + '\n' * 2)
    if not msg is None:
        print(msg)


def eachPlayerHasPath(board):
    global N, WALLS, G
    # heuristic when at most one wall
    nb_walls = board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2].sum()
    if nb_walls <= 1:
        # there is always a path when there is at most one wall
        return True
        # checks whether the two players can each go to the opposite side
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
        A = np.zeros((N, N), dtype='bool')  # TO DO: this could be optimized
        S = [coord[i]]  # set of nodes left to treat
        finished = False
        while len(S) > 0 and not finished:
            c = S.pop()
            # NB: In A we swap rows and columns for simplicity
            A[c[1]][c[0]] = True
            for k in range(4):
                if G[c[0]][c[1]][k] == 1:
                    s = steps[k]
                    new_c = c + s
                    # test whether we reached the opposite row
                    if i == 0:
                        if new_c[1] == N - 1:
                            finished = True
                            break
                    else:
                        if new_c[1] == 0:
                            return True
                    # otherwise we continue exploring
                    if A[new_c[1]][new_c[0]] == False:
                        # heuristic, we give priority to moves going up (down) in case player is white (black)
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
    # returns True if there is no wall in direction step from pos, and we stay in the board
    # NB: it does not check whether the destination is occupied by a player
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
            # move blocked by a wall
            return False
    return True


def computeGraph(board=None):
    global N, WALLS
    # order of steps in edge encoding: (1,0), (0,1), (-1,0), (0,-1)
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
                    # initial setup
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
        pos[i] += pn * N ** 2  # offset for black player
    P = []  # list of new boards (each encoded as list bits to switch)
    # current player moves to another position
    for s in steps:
        if canMove(board, coord[pn], s):
            new_coord = coord[pn] + s
            new_pos = pos[pn] + s[0] + N * s[1]
            occupied = np.array_equal(new_coord, coord[(pn + 1) % 2])
            if not occupied:
                P.append([pos[pn], new_pos])  # new board is obtained by switching these two bits
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
                            # current player puts down a wall
    # TO DO: Speed up this part: it would perhaps be faster to directly discard intersecting walls based on existing ones
    nb_walls_left = board[2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1):2 * N ** 2 + 2 * (N - 1) ** 2 + (pn + 1) * (
                WALLS + 1)].argmax()
    ind_walls_left = 2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1) + nb_walls_left
    if nb_walls_left > 0:
        for i in range(2 * (N - 1) ** 2):
            pos = 2 * N ** 2 + i
            L = [pos]  # indices of walls that could intersect
            if i < (N - 1) ** 2:
                # horizontal wall
                L.append(pos + (N - 1) ** 2)  # vertical wall on the same 4-square
                if i % (N - 1) > 0:
                    L.append(pos - 1)
                if i % (N - 1) < N - 2:
                    L.append(pos + 1)
            else:
                # vertical wall
                L.append(pos - (N - 1) ** 2)  # horizontal wall on the same 4-square
                if (i - (N - 1) ** 2) // (N - 1) > 0:
                    L.append(pos - (N - 1))
                if (i - (N - 1) ** 2) // (N - 1) < N - 2:
                    L.append(pos + (N - 1))
            nb_intersecting_wall = sum([board[j] for j in L])
            if nb_intersecting_wall == 0:
                board[pos] = 1
                # we remove the corresponding edges from G
                if i < (N - 1) ** 2:
                    # horizontal wall
                    a, b = i % (N - 1), i // (N - 1)
                    E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                else:
                    # vertical wall
                    a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                    E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                for e in E:
                    G[e[0]][e[1]][e[2]] = 0
                if eachPlayerHasPath(board):
                    P.append([pos, ind_walls_left - 1, ind_walls_left])  # put down the wall and adapt player's counter
                board[pos] = 0
                # we add back the two edges in G
                for e in E:
                    G[e[0]][e[1]][e[2]] = 1
                    # we create the new boards from P
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
    # player positions
    board[(N - 1) // 2] = True
    board[N ** 2 + N * (N - 1) + (N - 1) // 2] = True
    # wall counts
    for i in range(2):
        board[2 * N ** 2 + 2 * (N - 1) ** 2 + i * (WALLS + 1) + WALLS] = 1
    return board


def playGame(player1, player2, show=False, delay=0.0):
    global N, WALLS, G, G_INIT
    # initialization
    players = [player1, player2]
    board = startingBoard()
    G = G_INIT.copy()
    for i in range(2):
        players[i].color = i
    # main loop
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
                    # jeu en cours est humain contre IA, on affiche estimation probabilité de victoire pour blanc selon IA
                    p = forwardPass(board, players[i].NN)
                    msg += '\nEstimation IA : ' + "{0:.4f}".format(p)
                    msg += '\n'
            display(board, msg)
            time.sleep(delay)
        new_board = players[current_player].makeMove(board)
        # we compute changes of G (if any) to avoid recomputing G at beginning of listMoves
        # we remove the corresponding edges from G
        if not new_board is None:
            v = new_board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2] - board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2]
            i = v.argmax()
            if v[i] == 1:
                # a wall has been added, we remove the two corresponding edges of G
                if i < (N - 1) ** 2:
                    # horizontal wall
                    a, b = i % (N - 1), i // (N - 1)
                    E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                else:
                    # vertical wall
                    a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                    E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                for e in E:
                    G[e[0]][e[1]][e[2]] = 0
        board = new_board
        if board is None:
            # human player quit
            quit = True
            finished = True
        elif endOfGame(board):
            players[current_player].score += 1
            white_won = current_player == 0
            players[current_player].endGame(board, white_won)
            if show:
                display(board)
                time.sleep(0.3)
            finished = True
        else:
            current_player = (current_player + 1) % 2
    return quit

def train(NN, n_train=10000):
    if LS == 'Q-learning':
        learning_strategy1 = (LS, ALPHA)
        learning_strategy2 = (LS, ALPHA)
    elif LS == 'TD-lambda':
        learning_strategy1 = (LS, ALPHA, LAMB, np.zeros(NN[0].shape), np.zeros(NN[1].shape))
        learning_strategy2 = (LS, ALPHA, LAMB, np.zeros(NN[0].shape), np.zeros(NN[1].shape))
    agent1 = Player_AI(NN, EPS, learning_strategy1, 'agent 1')
    agent2 = Player_AI(NN, EPS, learning_strategy2, 'agent 2')
    # training session
    print('\nEntraînement ('+str(n_train)+' parties)' )
    for j in range(n_train):
        progressBar(j, n_train)
        playGame(agent1, agent2)

def main():
    global G_INIT

    NN = createNN(2 * 5 ** 2 + 2 * (5 - 1) ** 2 + 2 * (3 + 1), 40)

    G_INIT = computeGraph()

    train(NN, int(sys.argv[1]))

    np.savez(sys.argv[2], N=5, WALLS=3, W1=NN[0], W2=NN[1])

if __name__ == '__main__':
    main()



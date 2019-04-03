import os, sys
from select import select
from IA_partie2 import *


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
    return '  ' + str(int(100 * i / n)) + '%'


def listEncoding(board, N, WALLS):
    # outputs list encoding of board:
    # [ [i, j], [k, l], list_of_horizontal_walls, list_of_vertical_walls, walls_left_p1, walls_left_p2 ]
    # where [i, j] position of white player and [k, l] position of black player
    # and each wall in lists of walls is of the form [a, b] where [a,b] is the south-west square
    pos = [None,None]
    coord = [ [None,None], [None,None]]
    walls = [ [], [] ]
    walls_left = [ None, None ]
    for i in range(2):
        pos[i] = board[i*N**2:(i+1)*N**2].argmax()
        coord[i][0] = pos[i]%N
        coord[i][1] = pos[i]//N
        for j in range((N-1)**2):
            if board[2*N**2 + i*(N-1)**2 + j]==1:
                walls[i].append( [j%(N-1), j//(N-1)] )
        walls_left[i] = board[2*N**2 + 2*(N-1)**2 + i*(WALLS+1):2*N**2 + 2*(N-1)**2 + (i+1)*(WALLS+1)].argmax()
    return [ coord[0], coord[1], walls[0], walls[1], walls_left[0], walls_left[1] ]


def canMove(board, coord, step, N, WALLS):
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

def listMoves(board, current_player, N, WALLS,G):
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
        if canMove(board, coord[pn], s, N, WALLS):
            new_coord = coord[pn] + s
            new_pos = pos[pn] + s[0] + N * s[1]
            occupied = np.array_equal(new_coord, coord[(pn + 1) % 2])
            if not occupied:
                P.append([pos[pn], new_pos])  # new board is obtained by switching these two bits
            else:
                can_jump_straight = canMove(board, new_coord, s, N, WALLS)
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
                        if canMove(board, new_coord, d, N, WALLS):
                            final_pos = new_pos + d[0] + N * d[1]
                            P.append([pos[pn], final_pos])
                            # current player puts down a wall
    # TO DO: Speed up this part: it would perhaps be faster to directly discard intersecting walls based on existing ones
    nb_walls_left = board[
                    2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1):2 * N ** 2 + 2 * (N - 1) ** 2 + (pn + 1) * (
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
                if eachPlayerHasPath(board, N, G):
                    P.append(
                        [pos, ind_walls_left - 1, ind_walls_left])  # put down the wall and adapt player's counter
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

def endOfGame(board,N):
    return board[(N - 1) * N:N ** 2].max() == 1 or board[N ** 2:N ** 2 + N].max() == 1

def startingBoard(N, WALLS):
    board = np.array([0] * (2 * N ** 2 + 2 * (N - 1) ** 2 + 2 * (WALLS + 1)))
    # player positions
    board[(N - 1) // 2] = True
    board[N ** 2 + N * (N - 1) + (N - 1) // 2] = True
    # wall counts
    for i in range(2):
        board[2 * N ** 2 + 2 * (N - 1) ** 2 + i * (WALLS + 1) + WALLS] = 1
    return board


def eachPlayerHasPath(board, N, G):
    # heuristic when at most one wall
    nb_walls = board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2].sum()
    if nb_walls <= 1:
        # there is always a path when there is at most one wall
        return True
    # checks whether the two players can each go to the opposite side
    pos = [None,None]
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
# this is an implementation of a gomoku game on n*n go board
import numpy as np
# this is an implementation of a gomoku game on n*n go board
#import numpy as np
#prossible value of a state entry:0,1,2
#each move is describe in a number row*lengtg+col
# this is an implementation of a gomoku game on n*n go board
# this is an implementation of a gomoku game on n*n go board
#import numpy as np
#prossible value of a state entry:0,1,2
#each move is describe in a number row*lengtg+col
class gomoku(object):#state object
    def __init__(self, size,goal):
        self.width = size
        self.height = size
        self.size=size
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = goal
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]
    def rdo_move(self,move):
        rand_val=round(np.random.uniform(high=5))
        if rand_val>3:
            if move in self.availables:
                self.do_move(move)
        if rand_val==3:
            if self.location_to_move(self.move_to_location(move+self.width)) in self.availables:
                move=self.location_to_move(self.move_to_location(move+self.width))
                self.do_move(move)
        if rand_val==2:
            if self.location_to_move(self.move_to_location(move+1)) in self.availables:
                move=self.location_to_move(self.move_to_location(move+1))
                self.do_move(move)
                
        if rand_val==1:
            if self.location_to_move(self.move_to_location(move-self.width)) in self.availables:
                move = self.location_to_move(self.move_to_location(move-self.width))
                self.do_move(move)
                
        if rand_val==0:
            if self.location_to_move(self.move_to_location(move-1)) in self.availables:
                move = self.location_to_move(self.move_to_location(move-1))
                self.do_move(move)
                
        self.last_move = move
        return self.last_move
    def do_move(self, move,random=False):
        if not random:
            self.states[move] = self.current_player
            self.availables.remove(move)
            self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
            )
            self.last_move = move
        else:
            self.rdo_move(move)
        

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif len(self.availables)==0:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
    

class Game(object):#create a game start with a not finished board
    def __init__(self,board):
        self.board=board
                    
    def play(self, player1,player2,start_player=0, is_shown=1,random=False):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        turn=0
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        while True:
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                        print("turn ",turn)
                    else:
                        print("Game end. Tie")
                return winner,turn
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move= player_in_turn.action(self.board)
            self.board.do_move(move,random)
            #self.board.rdo_move(move)
            turn+=1
    def self_play_MCTS(self,player,is_shown=0,temp=1e-3,random=False):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.action(self.board,
                                                 temp=temp,
                                                 random=random)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move,random=random)
            #self.board.rdo_move(move)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie") 
                return winner, zip(states, mcts_probs, winners_z)

            
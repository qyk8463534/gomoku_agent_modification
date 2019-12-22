import numpy as np
import copy
from operator import itemgetter
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
class MCTSNode(object):#a node means a action
    def __init__(self, parent,P):
        self.parent=parent
        #self.move=None
        self.children={}#dictionary where [action]=node
        self.P=P#prior probability of selecting current move
        self.Q=0#mean value of next state
        self.W=0#total value of next state
        self.N=0#visit times of next state
    def is_leaf(self):
        return self.children=={}
    def is_root(self):
        return self.parent is None
    def value(self,para):
        U=(para * self.P *np.sqrt(self.parent.N) / (1 + self.N))
        #U=para*self.P*self.parent.N/(self.N+1.0)
        return self.Q+U
    def expand(self,action_list):#action_list=[(action,priority probability)]
        for a,p in action_list:
            if a not in self.children:
                self.children[a]=MCTSNode(self,p)
    def update(self,value):#value of leaf
        self.N+=1.0
        self.W+=value*1.0
        self.Q += 1.0*(value - self.Q) / self.N
        #self.Q=1.0*self.W/self.N
        #if self.parent == None:
        #    print('root')
        #    print([self.N,self.W,self.Q])
        #else:
        #    if self.parent.parent==None:
        #        print([self.N,self.W,self.Q])
    def select(self,para):#choose a node which maximize Q+U
        return max(self.children.items(),key=lambda node:node[1].value(para))
    def rupdate(self,value):#value of leaf
        if self.parent:
            self.parent.rupdate(-value)
        self.update(value)
class MCTS(object):
    def __init__(self,network,para=1.0,n_play=1600):
        self.root = MCTSNode(None,1.0)
        self.network = network
        self.para = para
        self.n_play=n_play

    def play(self, state,random=False):#state is a gomoku board instance;a single play
        '''node=self.root
        while not node.is_leaf():
            a,node=node.select(self.para)
            state.move(state.current_player,a//state.size,a%state.size)
            #state.score_after_move()
        p,v=self.network(state)#p is a list of probability
        end=state.is_finished()
        winner=state.winner
        if not end:
            node.expand(p)
        else:
            v=(1.0 if winner==state.get_current_player() else -1.0)
            node.rupdate(v)'''
        node = self.root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            a, node = node.select(self.para)
            #print(state.current_player)
            state.do_move(a,random)
            #state.rdo_move(a)
            
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self.network(state)
        # Check for end of game.
        end,winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        node.rupdate(-leaf_value)    
        
        # Update value and visit count of nodes in this traversal.

    def get_move_p(self,state,temp=1e-3,random=False):
        for i in range(self.n_play):
            state_rep=copy.deepcopy(state)
            self.play(state_rep,random)
        a_visits=[(a,node.N) for a,node in self.root.children.items()]
        a,visits=zip(*a_visits)
        a_p=softmax(1.0/temp*np.log(np.array(visits)+1e-10))
        return a,a_p
    def update_move(self,move):
        if move in self.root.children:
            self.root=self.root.children[move]
            self.root.parent=None
        else:
            self.root=MCTSNode(None,1.0)
class MCTS_Player(object):#a agent to play game
    def __init__(self,network,para=5,n_play=1600,is_selfplay=True):
        self.mcts=MCTS(network,para=para,n_play=n_play)
        self.is_selfplay=is_selfplay
    def set_player_symbol(self,symbol):
        self.player=symbol
    def reset(self):
        self.mcts.update_move(-1)
    def set_player_ind(self, p):
        self.player = p
    def action(self,board,temp=1e-3,random=False):
        available_moves=board.availables
        move_p=np.zeros(board.size**2)
        if len(available_moves)>0:
            a,p=self.mcts.get_move_p(board,temp,random)
            move_p[list(a)]=p
            if self.is_selfplay:
                #p_net,_=self.mcts.network(board)
                move = np.random.choice(a,p=0.75*p + 0.25*np.random.dirichlet(0.3*np.ones(len(p))))#P=0.75*P + 0.25*np.random.dirichlet(0.3*np.ones(len(P)))
                self.mcts.update_move(move)
                return move,move_p
            else:
                #p_net,_=self.mcts.network(board)
                move=np.argmax(move_p)
                self.reset()
                return move
        else:
            print("WARNING: the board is full")
            return None
    def __str__(self):
        return "MCTS? {}".format(self.player)

        

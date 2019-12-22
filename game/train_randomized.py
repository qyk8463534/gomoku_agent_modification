from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from gomoku import gomoku,Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts import MCTS_Player
from network import PolicyValueNet
from torch.utils.tensorboard import SummaryWriter
class train_network():
    def __init__(self,game_size=5,game_goal=3,randomized=False,model=None,folder_name='runs/test1'):
        self.size=game_size
        self.goal=game_goal
        self.randomized=randomized
        self.board=gomoku(self.size,self.goal)
        self.game=Game(self.board)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0
        self.n_play=40  
        self.para=5
        self.buffer_size=10000
        self.batch_size=64
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 100
        self.best_win_ratio = 0.0
        self.writer = SummaryWriter(folder_name)
        self.pure_mcts_playout_num = 40

        if model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.size,model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.size)
        self.mcts_player = MCTS_Player(self.policy_value_net.p_v_fn,para=self.para,n_play=self.n_play,is_selfplay=True)
    def symmetric_data(self,play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3,4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.size, self.size)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.self_play_MCTS(self.mcts_player,random=self.randomized)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.symmetric_data(play_data)
            self.data_buffer.extend(play_data)
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.p_v(state_batch)
        total_loss=0
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch,mcts_probs_batch,winner_batch,self.learn_rate*self.lr_multiplier)
            total_loss+=loss
            new_probs, new_v = self.policy_value_net.p_v(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return total_loss/self.epochs, entropy
    def policy_evaluate(self, n_games=10,model='./best_policy.model'):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTS_Player(self.policy_value_net.p_v_fn,para=self.para,n_play=self.n_play,is_selfplay=False)
        baseline_player=randomPlayer()
        pure_mcts_player = MCTS_Pure(c_puct=5,n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        total_turn=0
        for i in range(n_games):
            winner,turn = self.game.play(current_mcts_player,pure_mcts_player,start_player=i % 2,random=self.randomized)
            #winner,turn= self.game.play(current_mcts_player,baseline_player,start_player=i % 2)
            #winner,turn= self.game.play(pure_mcts_player,baseline_player,start_player=i % 2)
            win_cnt[winner] += 1
            total_turn+=turn
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[2]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio,total_turn

    def run(self):
        """run the training pipeline"""
        print('run')
        self.policy_value_net.save_model('./best_policy.model')
        total_turn=0
        try:
            for i in range(self.game_batch_num):
                print('collect data')
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print('writet write')
                    self.writer.add_scalar('training loss',loss,i)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio,turn= self.policy_evaluate(model='./best_policy.model')
                    total_turn=turn
                    print('total turn',total_turn)
                    self.writer.add_scalar('winning_rate',win_ratio,i+1)
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        #print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 40
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')
class randomPlayer(object):
    def __init__(self):
        pass
    def action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = random.choice(sensible_moves)
            return move
        else:
            print("WARNING: the board is full")
    def set_player_ind(self, p):
        self.player = p
    def __str__(self):
        return "BASE {}".format(self.player)
if __name__ == '__main__':
    training_pipeline = train_network(game_size=5,game_goal=3,randomized=True,folder_name="runs/64")
    training_pipeline.run()
    training_pipeline = train_network(game_size=5,game_goal=3,randomized=True,folder_name="runs/53")
    training_pipeline.run()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Net(nn.Module):
    '''
    def __init__(self,size):
        super(Net,self).__init__()
        self.size=size
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)#4 different direction
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #action probabilities
        self.act_conv1 = nn.Conv2d(128, 256, kernel_size=3)
        self.act_finalconv=nn.Conv2d(256, size**2, kernel_size=1)
        #state value
        self.val_conv1 = nn.Conv2d(128, 256, kernel_size=3)
        #self.val_conv2 = nn.Conv2d(256, 512, kernel_size=3)
        self.val_finalconv=nn.Conv2d(256, 1, kernel_size=1)
    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = F.adaptive_avg_pool2d(x_act, (1, 1))
        x_act = F.log_softmax(self.act_finalconv(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        #x_val = F.relu(self.val_conv2(x))
        x_val = F.adaptive_avg_pool2d(x_val, (1, 1))
        x_val = torch.tanh(self.val_finalconv(x_val))
        return x_act, x_val
       ''' 
    def __init__(self, size):
        super(Net, self).__init__()

        self.board_width = size
        self.board_height = size
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val
class PolicyValueNet():
    def __init__(self,size,model=None):
        self.size=size
        self.decay=1e-4

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net=Net(size).float().to(self.device)

        self.optimizer=optim.Adam(self.net.parameters(),weight_decay=self.decay)
        if model!=None:
            self.net.load_state_dict(torch.load(model))
    def p_v(self,batch):
        batch=Variable(torch.FloatTensor(batch).to(self.device))
        log_a_p,v=self.net(batch)
        #log_a_p=log_a_p.squeeze()
        a_p=np.exp(log_a_p.data.cpu().numpy())
        #v=v.squeeze().cpu()
        return a_p, v.data.cpu().numpy()
    def p_v_fn(self,state):
        legal_positions = state.availables
        current_state = np.ascontiguousarray(state.current_state().reshape(
                -1, 4, self.size, self.size))
        log_act_probs, value = self.net(Variable(torch.from_numpy(current_state)).float().to(self.device))
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value=value.squeeze().cpu()
        value= value.item()
        return act_probs, value
    def train_step(self, state_batch, mcts_probs, winner_batch,lr):
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
        #j=0
        #for i in mcts_probs:
        #    print(j)
        #    print(i)
        #    j+=1
        winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.zero_grad()
        log_act_probs, value=self.net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))/(self.size*np.log(self.size)/2)
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(),entropy
    def get_policy_param(self):
        net_params = self.net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


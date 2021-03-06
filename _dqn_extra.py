import replaybuffer
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def to_torch_var(x, vol=False):
    if vol:
        return Variable(torch.Tensor(x), volatile=True)
    else:
        return Variable(torch.Tensor(x))

class Q_Network(nn.Module):
    def __init__(self, state_dim , action_dim,xw_size):
        super(Q_Network, self).__init__()
        self.x_layer = nn.Linear(state_dim, xw_size)
        self.h_layer = nn.Linear(xw_size, xw_size)
        self.y_layer = nn.Linear(xw_size, action_dim)

    def forward(self, x):
        x = F.relu(self.x_layer(x))
        x = F.relu(self.h_layer(x))
        state_values = self.y_layer(x)
        return state_values

class DeepQNetwork(object):
    def __init__(self, state_dim, action_dim):
        self.replay_buffer = replaybuffer.ReplayBuffer()
        self.q_network = Q_Network(state_dim, action_dim,16)
        self.q_network_target = copy.deepcopy(self.q_network)
        self.q_network_optim = torch.optim.Adam(
            self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = 0.995
        self.tau = 0.05

    def get_policy_action(self, state):
        state = to_torch_var(state.reshape(-1, self.state_dim))
        return np.argmax(self.q_network(state).cpu().data.numpy().flatten())

    def soft_target_update(self,network,target_network,tau):
        for net_params, target_net_params in zip(network.parameters()
                , target_network.parameters()):
            target_net_params.data.copy_(
                net_params.data * tau + target_net_params.data * (1 - tau))

    def update_Q_Network(self, state, next_state, action, reward, done_bool):
        with torch.no_grad():
            qsa_next_action = self.q_network_target(next_state)
            qsa_next_action = torch.max(qsa_next_action, dim=1)
            qsa_next_action = qsa_next_action[0].unsqueeze(1)
            qsa_next_target = reward + done_bool * self.discount * qsa_next_action
        qsa = self.q_network(state)
        qsa = qsa * action
        qsa = torch.sum(qsa, dim=1).unsqueeze(1)
        q_network_loss = self.criterion(qsa, qsa_next_target)
        self.q_network_optim.zero_grad()
        q_network_loss.backward()
        self.q_network_optim.step()
        pass

    def update(self, update_rate):
        for i in range(update_rate):
            batch = self.replay_buffer.sample_batch(256)
            state = to_torch_var(np.array(batch[0]))
            next_state = to_torch_var(np.array(batch[1]))
            action = to_torch_var(np.array(batch[2]))
            reward = to_torch_var(np.array(batch[3]).reshape(-1, 1))
            done_bool = 1 - to_torch_var(np.array(batch[4]).reshape(-1, 1))
            self.update_Q_Network(state, next_state, action, reward, done_bool)
            self.soft_target_update(self.q_network, self.q_network_target, self.tau)
from model import SeqRosModel
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time,sys
import matplotlib.pyplot as plt
# Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 8000
env = SeqRosModel()

N_ACTIONS = env.n_actions
N_STATES = env.n_observations

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

np.random.seed(2)
torch.manual_seed(2)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(512, 1024)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization 
        self.fc3 = nn.Linear(1024, 1024)
        self.fc3.weight.data.normal_(0, 0.1)   # initialization                
        self.out = nn.Linear(1024, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        # x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        if use_cuda:
            self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        else:
            self.eval_net, self.target_net = Net(), Net()
        self.e_greedy = EPSILON
        self.learning_rate = LR
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        
        if use_cuda:
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0).cuda())
        else:
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < self.e_greedy:   # greedy
            actions_value = self.eval_net.forward(x)
            if use_cuda:
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax

        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('Parameters updated')
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        if use_cuda:
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]).cuda())
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).cuda())
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda())
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]).cuda())
        else:
            
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu().numpy()

dqn = DQN()
fig1 = plt.figure(1)

plt.ion()
episode_list = []
reward_list = []
reward_list_print = []
loss_list = []
episode_loss_list = []
reward_draw = 0
loss = -1
loss_total = 0

action_value_list = []
episode_action_value_list = []
action_value_total = 0


print('\nCollecting experience...')

for i_episode in range(1101):
    if i_episode % 150 == 149:
        dqn.e_greedy += 0.05
        if dqn.e_greedy > 0.95:
            dqn.e_greedy = 0.95

    s = env.reset()
    ep_r = 0
    counter = 0

    while True:
        if i_episode % 1000 == 0:
            name = 'dqn_eval_net_' + str(i_episode) + '.pkl'
            torch.save(dqn.eval_net.state_dict(), name)

        a = dqn.choose_action(s)

        # take action
        s_, r, done = env.step(a)

        dqn.store_transition(s, a, r, s_)
        counter += 1
        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            loss = dqn.learn()
            if done:
                env.render()
                print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',ep_r)

        if done:
            break
        s = s_

    reward_draw += ep_r

    if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY+220:
        reward_list_print.append(reward_draw/10.0)

    if i_episode % 10 == 0:
        episode_list.append(i_episode)
        reward_list.append(reward_draw/10.0)
        action_value_list.append(action_value_total/10.0)
        plt.figure(1)
        plt.cla()
        plt.plot(episode_list, reward_list, label='Reward')
        plt.xlabel('Training Epochs')
        plt.ylabel('Reward')
        plt.draw()
        plt.pause(0.1)

        reward_draw = 0
        action_value_total = 0

        if i_episode % 1000 == 0 and i_episode > 0:
            fig1.savefig('fig_reward_'+str(i_episode)+'.eps', format='eps', dpi=fig1.dpi)


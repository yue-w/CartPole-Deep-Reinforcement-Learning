# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 09:35:12 2020

@author: wyue
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import gym
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output


class DQN(nn.Module):
    """
    input: state
    output: Q-values corresponging to the two actions
    """
    def __init__(self, state_dim,action_dim, hidden_dim):
        super(DQN,self).__init__()
        #self.device = device
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2,action_dim)
        
    def forward(self,state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        q_values = self.fc3(h2)
        
        return q_values

class Agent():
    def __init__(self,dqn,epsilon,capacity,gamma,minibatch_size,device):
        ## Two neural networks with the same structures are used
        ## The fist network dqn is used to make actions (Q in the paper)
        ## The second network is used to compute target (Q hat in the paper).
        ## dqn is cloned to dqn_target for every C (a given constant) steps
        self.dqn = dqn
        self.dqn_target = copy.deepcopy(self.dqn)
        ## memory is used for replay
        self.memory = []
        ## The capacity of memory
        self.capacity = capacity
        self.epsilon = epsilon
        ## gamma is the depreciation of value with regard to timestep
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.device = device
        
    
    def update_memory(self,state, action, next_state, reward, done):
        new_mem = (state, action, next_state, reward, done)
        ## If less than memory capacity, add to memory
        if len(self.memory) < self.capacity:
            self.memory.append(new_mem)
        ## if memory full, randomly replace old memory with new memory
        else:
            position = random.randint(0,self.capacity-1)
            self.memory[position] = new_mem


    def q_values(self,state):
        """ Compute Q values for all actions using the DQN. """
        with torch.no_grad():
            return self.dqn(state)
        
    def update_parameters(self):
        ''' Update target network with the model weights.'''
        self.dqn_target.load_state_dict(self.dqn.state_dict())


    def take_action(self, environment, state):
        if random.random() < self.epsilon:
            #action = random.sample([0,1],1)[0]
            action = environment.action_space.sample()
        else:
            state = torch.from_numpy(state).float().to(self.device)
            q_all = self.q_values(state)
            action = torch.argmax(q_all).item()
            ## Cast the state tensor back to numpy
            state = state.cpu().data.numpy()
        ## Take action and simulate in the environment, record reward, next state, 
        ## and whether terminate (whether done==True)
        next_state, reward, done, _ = environment.step(action)
        self.update_memory(state, action, next_state, reward, done)
        return action, next_state, reward, done
    
        
    def replay(self):

        batch = random.sample(self.memory, self.minibatch_size)
        ## Resctucture the minibatch in the shape: [[state minibatch],[action minibatch] ... ]
        batch_t = list(map(list, zip(*batch)))
        #batch_t_tensor = torch.Tensor(batch_t)
        is_done_batch = torch.Tensor(batch_t[4])
        
        ##state_batch_NotDone = state_batch[np.logical_not(is_done_batch)]

        not_done_indices = torch.where(is_done_batch==False)[0]
        done_indices = torch.where(is_done_batch==True)[0]
        
        batch_size = len(not_done_indices)
        state_batch = torch.zeros((batch_size, 4)).to(self.device)
        action_batch = torch.zeros((batch_size,1),dtype=int).to(self.device)
        next_state_batch = torch.zeros((batch_size, 4)).to(self.device)
        reward_batch = torch.zeros((batch_size,1)).to(self.device)
        
        for i,index in enumerate(not_done_indices):
            state_batch[i] = torch.Tensor(batch_t[0][index])
            action_batch[i] = torch.Tensor([batch_t[1][index]])# dtype=torch.int8
            next_state_batch[i] = torch.Tensor(batch_t[2][index])
            reward_batch[i] = torch.Tensor([batch_t[3][index]])

        q_values = self.q_values(next_state_batch)
        qmax = torch.max(q_values,dim=1).values
        qmax = qmax.reshape((batch_size,1))
        
        target = reward_batch + self.gamma * qmax

        batch_size_done = len(done_indices)

        if batch_size_done>0:## The state 'Done' is rare, we put it in if
            #y_batch_done = np.zeros((done_indices,1))
            state_batch_done = torch.zeros((batch_size_done, 4)).to(self.device)
            action_batch_done = torch.zeros((batch_size_done, 1),dtype=int).to(self.device)
            #next_state_batch_done = np.zeros((batch_size_done, 4))
            reward_batch_done = torch.zeros((batch_size_done, 1)).to(self.device)
            for i,index in enumerate(done_indices):
                state_batch_done[i] = torch.Tensor(batch_t[0][index])
                action_batch_done[i] = torch.Tensor([batch_t[1][index]])
                #next_state_batch_done[i] = batch_t[2][index]
                reward_batch_done[i] = torch.Tensor([batch_t[3][index]])
            
            ## Append the cases of "Done"
            target = torch.cat((target,reward_batch_done),0)
            state_batch = torch.cat((state_batch,state_batch_done),0)
            action_batch = torch.cat((action_batch,action_batch_done),0)
        
        ## Compute Q values of all actions
        q_batch= self.dqn(state_batch)
        ## Use the Q-value corresponding to (state,action) pair. (Not necessary the largest one)
        y_pred = torch.zeros_like(target).to(self.device)
        for i in range(len(y_pred)):
            y_pred[i] = q_batch[i][action_batch[i]].clone()

        return target, y_pred

def plot_res(values, title=''):
    """
    This function is copied from Rita Kurban's tutorial:
    https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
    """
    ## Plot the reward curve and histogram of results over time.
    # Update the window after each episode
    clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    ax[0].legend()
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

def print_running_info(i,total_reward,next_state):
    print("episode: {}, total reward: {}".format(i, total_reward))
    x = next_state[0]
    theta = next_state[2]
    if x <= -2.4 or x >= 2.4:
        print('Out of screen')
    elif theta < -0.20944 or theta > 0.20944:
        print('Pole fall')
    else:
        '????'

            
def train(environment,state_dim, action_dim, hidden_dim, lr,replaycapacity, episodes,device, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99, minibatch_size=64, n_update=10):
    criterion = torch.nn.MSELoss()
    dqn = DQN(state_dim,action_dim, hidden_dim)
    agent = Agent(dqn,epsilon,replaycapacity,gamma,minibatch_size,device)
    agent.dqn.to(agent.device)
    agent.dqn_target.to(agent.device)
    optimizer = torch.optim.Adam(agent.dqn.parameters(), lr)
    
    total_reward_history = []
    for episode in range(1,episodes):
        if episode%150==0:
            lr /= 2
            agent.optimizer = torch.optim.Adam(agent.dqn.parameters(), lr)
        # Every C steps, reset target network (Q hat) to the action network (Q)
        if episode % n_update == 0:
            agent.update_parameters()
        
        # Reset state
        state = environment.reset()
        done = False
        total_reward = 0

        while not done:
            action, next_state, reward, done = agent.take_action(environment, state)
            total_reward += reward
            agent.update_memory(state, action, next_state, reward, done)
            ## Replay only if the size of the memory is larger than the minibatch
            if agent.minibatch_size <= len(agent.memory):
                target, y_pred = agent.replay()
                loss = criterion(target, y_pred) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
        # Update epsilon
        agent.epsilon = max(epsilon * eps_decay, 0.1)
        total_reward_history.append(total_reward)
        plot_res(total_reward_history)
        print_running_info(episode,total_reward,next_state)
    
    return agent

environment = gym.envs.make("CartPole-v1")

# Number of states
n_state = environment.observation_space.shape[0]
# Number of actions
n_action = environment.action_space.n
# Number of episodes
episodes = 500
# Number of hidden nodes in the DQN
n_hidden = 64
# Learning rate
lr = 0.00025
replaycapacity = 100000

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
train(environment,n_state, n_action, n_hidden, lr,replaycapacity, episodes,device, gamma=0.99,epsilon=0.2,eps_decay=0.9)

import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import sys 
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../src/agents'))
from src.agents.base_agent import BaseAgent  
from src.env_sailing import SailingEnv
from src.initial_windfields import get_initial_windfield, INITIAL_WINDFIELDS


device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu"
                    )
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent(BaseAgent):
    def __init__(self):
        self.envs = [SailingEnv(**get_initial_windfield("simple_static")),
                    SailingEnv(**get_initial_windfield("training_1")),
                     SailingEnv(**get_initial_windfield("training_2")),
                     SailingEnv(**get_initial_windfield("training_3"))]
        self.env = SailingEnv(**get_initial_windfield("training_1"))
        state, _ = self.env.reset()
        self.n_observations = 6 #len(state)  #######
        self.n_actions = 8
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []

        self.BATCH_SIZE = 256
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 5000
        self.TAU = 0.005
        # self.reset()

    def seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)

    def reset(self):
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.policy_net = self.load("src/agents/models/dqn_trained.pth")
        obs, _ = self.env.reset()
        return obs

    def act(self, observation):
        observation = observation[:6]  #######
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        #if random.random() > eps_threshold:
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.item()
        #else:
            #return self.env.action_space.sample()
        
    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)


    def train(self, num_episodes):
        for env in self.envs:
            for i_episode in tqdm(range(num_episodes)):
            #env = self.env
                state, _ = env.reset(seed=i_episode)
                state = state[:6]  #######
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                episode_reward = 0

                for t in count():
                    action = self.act(state.squeeze().cpu().numpy())
                    observation, reward, terminated, truncated, _ = env.step(action)
                    observation = observation[:6]  #######
                    reward = torch.tensor([reward], device=device)
                    episode_reward += reward 
                    done = terminated or truncated

                    if done:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                    self.memory.push(state, torch.tensor([[action]], device=device), next_state, reward)
                    state = next_state

                    self.optimize_model()
                    self._soft_update_target_network()

                    if done:
                        print("Episode duration:", t+1)
                        self.episode_durations.append(t + 1)
                        self.episode_rewards.append(episode_reward)
                        break
        return self.episode_durations

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

    def plot_duration(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Result' if show_result else 'Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="src/agents/models/dqn_trained.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.policy_net.to(device)
        return self.policy_net.to(device)
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange
import numpy as np
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import random
import os
import sys
import matplotlib.pyplot as plt 

sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('..'))

from src.agents.base_agent import BaseAgent
from src.initial_windfields import get_initial_windfield, INITIAL_WINDFIELDS
#from src.env_sailing import SailingEnv


class ReinforceAgent(BaseAgent):
    """Policy Gradient Method - REINFORCE """
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        print("Reinforce class loaded")  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_s = 2054
        self.n_actions = 8
        self.gamma = 0.99
        self.lr = 0.001
        self.nb_episodes = 10
        self.policy = PolicyNetwork(input_dim=self.d_s, output_dim=self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=2**-13)    

    def act(self, observation):
        state_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        action = self.policy.sample_action(state_tensor)
        return action
    
    def gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        states = []
        actions = []
        returns = []
        for ep in tqdm(range(self.nb_episodes)):
            x,_ = env.reset(seed=ep)
            rewards = []
            episode_cum_reward = 0
            time = 0
            while(True):
                a = self.policy.sample_action(torch.as_tensor(x))
                y,r,done,trunc,_ = env.step(a)
                states.append(x)
                actions.append(a)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                #if time % 100==0:
                    #print("Current time: ", time)
                    #self.plot(x)
                time += 1
                if done: 
                    #print("Done!")
                    new_returns = []
                    G_t = 0
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
                
        # make loss
        returns = torch.tensor(returns)
        log_prob = self.policy.log_prob(torch.as_tensor(np.array(states)),torch.as_tensor(np.array(actions)))
        loss = -(returns * log_prob).mean()
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def reset(self):
        pass

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, nb_rollouts):
        avg_sum_rewards = []
        for ep in trange(nb_rollouts):
            for initial_windfield_name, initial_windfield in INITIAL_WINDFIELDS.items():
                env = SailingEnv(**get_initial_windfield(initial_windfield_name))
                avg_sum_rewards.append(self.gradient_step(env))
        return avg_sum_rewards
    
    def plot(self, observation):
        plt.figure(figsize=(10, 10))
        img = SailingEnv.visualize_observation(observation)  
        plt.imshow(img)
        plt.axis('off')
        plt.title('Environment Visualization from Observation')
        plt.show()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"✅ Agent saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy.eval()
        print(f"🔁 Agent loaded from {path}")

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        state_dim = 2054
        n_action = 8
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, n_action)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)
    
    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)
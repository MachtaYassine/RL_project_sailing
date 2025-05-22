import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from env_sailing import SailingEnv
from agents.Deterministic_agent2 import DeterministicAgent2
from initial_windfields import INITIAL_WINDFIELDS

# --- GAIL Components ---
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, obs):
        return self.net(obs)
    def act(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

def collect_expert_trajectories(agent, windfield, num_episodes=200, max_steps=100):
    X, y = [], []
    for ep in tqdm(range(num_episodes), desc="Collecting expert data for GAIL"):
        env = SailingEnv(
            wind_init_params=windfield['wind_init_params'],
            wind_evol_params=windfield['wind_evol_params']
        )
        obs, _ = env.reset(seed=ep)
        agent.reset()
        for _ in range(max_steps):
            action = agent.act(obs)
            X.append(obs.copy())
            y.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
    return list(zip(X, y))

def train_gail(env_fn, expert_trajectories, obs_dim, action_dim, num_epochs=1000, batch_size=64):
    policy = PolicyNet(obs_dim, action_dim)
    discriminator = Discriminator(obs_dim, 1)
    optim_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optim_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    expert_obs, expert_acts = zip(*expert_trajectories)
    expert_obs = torch.tensor(np.array(expert_obs), dtype=torch.float32)
    expert_acts = torch.tensor(np.array(expert_acts), dtype=torch.float32).unsqueeze(1)
    for epoch in range(num_epochs):
        # 1. Sample policy trajectories
        env = env_fn()
        obs, _ = env.reset()
        traj_obs, traj_acts = [], []
        for _ in range(batch_size):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act, _ = policy.act(obs_tensor)
            act = act.item()
            traj_obs.append(obs)
            traj_acts.append([act])
            obs, _, done, truncated, _ = env.step(act)
            if done or truncated:
                obs, _ = env.reset()
        traj_obs = torch.tensor(np.array(traj_obs), dtype=torch.float32)
        traj_acts = torch.tensor(np.array(traj_acts), dtype=torch.float32)
        # 2. Train Discriminator
        expert_idx = torch.randint(0, len(expert_obs), (batch_size,))
        e_obs = expert_obs[expert_idx]
        e_acts = expert_acts[expert_idx]
        d_real = discriminator(e_obs, e_acts)
        d_fake = discriminator(traj_obs.detach(), traj_acts.detach())
        loss_disc = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
        optim_disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()
        # 3. Train Policy (REINFORCE-style update)
        d_fake = discriminator(traj_obs, traj_acts)
        rewards = -torch.log(1 - d_fake + 1e-8).squeeze()
        log_probs = []
        for ob, ac in zip(traj_obs, traj_acts):
            _, log_prob = policy.act(ob.unsqueeze(0))
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs).squeeze()
        loss_policy = -torch.mean(log_probs * rewards)
        optim_policy.zero_grad()
        loss_policy.backward()
        optim_policy.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: D_loss={loss_disc.item():.4f}, Policy_loss={loss_policy.item():.4f}")
    return policy

def evaluate_policy(policy, windfield_names, max_steps=200, episodes_per_wf=10, device="cpu"):
    """
    Evaluate a trained policy on a set of windfields.
    Returns:
        success_rates: List of success rates per windfield
        avg_steps: List of average steps per windfield
    """
    policy.eval()
    success_rates = []
    avg_steps = []
    for windfield_name in windfield_names:
        windfield = INITIAL_WINDFIELDS[windfield_name]
        successes = []
        steps_list = []
        for ep in range(episodes_per_wf):
            env = SailingEnv(
                wind_init_params=windfield['wind_init_params'],
                wind_evol_params=windfield['wind_evol_params']
            )
            obs, _ = env.reset(seed=ep)
            steps = 0
            success = False
            for _ in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = policy.act(obs_tensor)
                action = action.item()
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                if done:
                    success = True
                    break
                if truncated:
                    break
            successes.append(success)
            steps_list.append(steps)
        success_rate = np.mean(successes)
        avg_step = np.mean(steps_list)
        success_rates.append(success_rate)
        avg_steps.append(avg_step)
    policy.train()
    return success_rates, avg_steps

if __name__ == "__main__":
    # Setup
    windfield = INITIAL_WINDFIELDS["training_1"]  # You can loop over more windfields
    expert_agent = DeterministicAgent2()
    expert_data = collect_expert_trajectories(expert_agent, windfield, num_episodes=200, max_steps=100)
    env_fn = lambda: SailingEnv(wind_init_params=windfield['wind_init_params'], wind_evol_params=windfield['wind_evol_params'])
    obs_dim = expert_data[0][0].shape[0]
    action_dim = 9  # 9 discrete actions
    policy = train_gail(env_fn, expert_data, obs_dim, action_dim, num_epochs=1000, batch_size=64)

    # --- Evaluation ---
    VAL_WF = ["training_6", "training_7", "training_8"]
    success_rates, avg_steps = evaluate_policy(policy, VAL_WF, max_steps=200, episodes_per_wf=10)
    print("Validation success rates:", success_rates)
    print("Validation avg steps:", avg_steps)

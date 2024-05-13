import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Parameters
gamma = 0.99  # Discount factor for future rewards
learning_rate = 0.01

env = gym.make('FrozenLake-v1', is_slippery=False)  # Non-slippery version for simplicity

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(env.observation_space.n, env.action_space.n)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train(policy, optimizer, log_probs, rewards):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + gamma**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)  # Normalize

    policy_loss = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * Gt)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

for episode in range(1000):  # Run for a certain number of episodes
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    
    while not done:
        state = torch.tensor([state], dtype=torch.float32)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        state, reward, done, _ = env.step(action.item())
        
        log_probs.append(m.log_prob(action))
        rewards.append(reward)
    
    train(policy, optimizer, log_probs, rewards)

    if episode % 50 == 0:
        print(f'Episode {episode} complete')


total_rewards = 0
num_episodes = 100

for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        state = torch.tensor([state], dtype=torch.float32)
        probs = policy(state)
        action = torch.argmax(probs).item()  # Always choose the best action
        state, reward, done, _ = env.step(action)
        if done and reward == 1:
            total_rewards += 1

print(f"Success rate: {total_rewards / num_episodes}")

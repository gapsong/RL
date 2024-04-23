import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
import torch

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
test = np.vstack((a,b))
print(test)
test2 = np.stack((a,b), axis=2)
print(test2)

num_inputs = 4
num_actions = 2

model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_actions, bias=False, dtype=torch.float32),
    torch.nn.Softmax(dim=1)
)

# Create the CartPole environment
env = gym.make('CartPole-v1')
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
# Initialize the environment and get the initial observation


def run_episode(max_steps_per_episode=10000, render=False):
    states, actions, probs, rewards = [], [], [], []
    state = env.reset()[0]
    for k in range(max_steps_per_episode):
        if render:
            env.render()
        action_probs = model(torch.from_numpy(np.expand_dims(state, 0)))[0]
        action = np.random.choice(
            num_actions, p=np.squeeze(action_probs.detach().numpy()))
        nstate, reward, done, info, random = env.step(action)
        if done:
            print(k, 'times')
            break
        states.append(state)
        actions.append(action)
        probs.append(action_probs.detach().numpy())
        rewards.append(reward)
        state = nstate
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)

s, a, p, r = run_episode()
print(f"Total reward: {np.sum(r)}")
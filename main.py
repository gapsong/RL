import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
import torch
import grid_world as gw


num_inputs = 4
num_actions = 2

model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_actions, bias=False, dtype=torch.float32),
    torch.nn.Softmax(dim=1)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create the CartPole environment
env = gym.make('CartPole-v1')
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
# Initialize the environment and get the initial observation


def play_until_lose(max_steps_per_episode=10000):
    states, actions, probs, rewards = [], [], [], []
    state = env.reset()[0]
    for k in range(max_steps_per_episode):
        action_probs = model(torch.from_numpy(
            np.expand_dims(state, 0)).float().to(device))[0]
        action = np.random.choice(
            num_actions, p=np.squeeze(action_probs.detach().cpu().numpy()))
        nstate, reward, done, info, random = env.step(action)
        if done:
            print(k, 'times')
            break
        states.append(state)
        actions.append(action)
        probs.append(action_probs.detach().cpu().numpy())
        rewards.append(reward)
        state = nstate
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


eps = 0.0001


def discounted_rewards(rewards, gamma=0.99, normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    return ret


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train_on_batch(x, y):
    # ist ein Batch von meinen verschiedenen steps
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions) * y)
    loss.backward()
    optimizer.step()
    return loss


alpha = 1e-4

history = []
for epoch in range(300):
    states, actions, probs, rewards = play_until_lose()
    one_hot_actions = np.eye(2)[actions.T][0]
    gradients = one_hot_actions-probs
    dr = discounted_rewards(rewards)
    gradients *= dr
    delta = alpha * np.vstack([gradients])
    target = delta + probs
    train_on_batch(states, target)
    history.append(np.sum(rewards))
    if epoch % 100 == 0:
        print(f"{epoch} -> {np.sum(rewards)}")

plt.plot(history)


_ = play_until_lose()

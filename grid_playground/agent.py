import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from grid_world import GridWorldEnv, MAPS

num_inputs = 4
num_actions = 4


class GridWorldModel(nn.Module):
    def __init__(self, grid_size, output_size):
        super(GridWorldModel, self).__init__()
        # Input layer the world
        self.fc1 = nn.Linear(2 * 2, 128)
        self.fc2 = nn.Linear(128, 128)    # Hidden layer
        self.fc3 = nn.Linear(128, output_size)  # Output layer

    def forward(self, x):
        x = torch.F.relu(self.fc1(x))  # Activation function ReLU
        x = torch.F.relu(self.fc2(x))  # Activation function ReLU
        x = self.fc3(x)
        return x


model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_actions, bias=False, dtype=torch.float32),
    torch.nn.Softmax(dim=1)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create the CartPole environment
env = GridWorldEnv(obstacle_map=MAPS['4x4'], render_mode="human")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
# Initialize the environment and get the initial observation


def play_until_lose(max_steps_per_episode=10000):
    states, actions, probs, rewards = [], [], [], []
    state = env.reset()
    for k in range(max_steps_per_episode):
        input_tensor = torch.cat(
            (state.agent, state.target)).float().unsqueeze(0)  # Prepare input
        input_tensor.to(device)
        action_probs = model(input_tensor)
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

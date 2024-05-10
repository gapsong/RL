import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from grid_world import GridWorldEnv

num_inputs = 4
num_actions = 4


class GridWorldModel(nn.Module):
    def __init__(self, input_channels, output_size):
        super(GridWorldModel, self).__init__()
        # Define convolutional layers
        # Example: 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        # Example: 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Example of an Adaptive Pooling layer to make sure the output is a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer
        # Assuming the last conv layer outputs 64 channels
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU after first conv layer
        x = F.relu(self.conv2(x))  # Apply ReLU after second conv layer
        x = self.adaptive_pool(x)  # Adaptive pooling to reduce to 1x1
        x = x.squeeze()
        # Flatten the tensor for the fully connected layer
        x = self.fc(x)  # Output layer
        x = F.softmax(x, dim=0)

        return x


model = GridWorldModel(input_channels=1, output_size=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create the CartPole environment
env = GridWorldEnv()
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
# Initialize the environment and get the initial observation


def play_until_lose(max_steps_per_episode, device):
    states, actions, probs, rewards = [], [], [], []
    state = env.reset()[0]
    for k in range(max_steps_per_episode):
        level = state['level']
        input_tensor = torch.tensor(level).type(torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        action_probs = model(input_tensor)

        action = np.random.choice(
            num_actions, p=np.squeeze(action_probs.detach().cpu().numpy()))
        nstate, reward, done, info, random = env.step(action)
        states.append(level)
        actions.append(action)
        probs.append(action_probs.detach().cpu().numpy())
        rewards.append(reward)
        state = nstate
        if done:
            print(k, 'times', np.sum(rewards))

            break
    return np.stack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


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


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_on_batch(x, y, device):
    # istrain_on_batcht ein Batch von meinen verschiedenen steps
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    optimizer.zero_grad()

    input_tensor = x.clone().detach().type(torch.float32)
    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = torch.tensor(input_tensor)

    predictions = model(input_tensor)
    loss = -torch.mean(torch.log(predictions) * y)
    loss.backward()
    optimizer.step()
    return loss


alpha = 1e-4

history = []
for epoch in range(600):
    states, actions, probs, rewards = play_until_lose(
        max_steps_per_episode=10000, device=device)
    one_hot_actions = np.eye(num_actions)[actions.T][0]
    gradients = one_hot_actions-probs
    dr = discounted_rewards(rewards, normalize=True)
    gradients *= dr
    delta = alpha * np.vstack([gradients])
    target = delta + probs
    train_on_batch(states, target, device)
    history.append(np.sum(rewards))
    if epoch % 100 == 0:
        print(f"{epoch} -> {np.sum(rewards)}")

plt.plot(history)
print('end')

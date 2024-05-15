import numpy as np
import random
import matplotlib.pyplot as plt

# Q-learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, state_size, action_size))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Exploration
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Exploitation

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.gamma * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.learning_rate * td_error

# Main training loop
def train(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes} - Total Reward: {total_reward}")
    return rewards

# Initialize environment and agent
size = 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]
env = GridEnvironment(size, start, goal, obstacles)
agent = QLearningAgent(state_size=size, action_size=4)

# Train the agent
episodes = 1000
rewards = train(env, agent, episodes)

# Plot rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Render the final policy
env.reset()
env.render()

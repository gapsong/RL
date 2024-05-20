import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from grid_world import GridWorldEnv

matplotlib.use('Qt5Agg')

epsilon = 0.3
q_table = np.zeros((5 * 5, 4))
env = GridWorldEnv()
observation, info = env.reset()
alpha = 0.1
gamma = 0.1
done = False
state = observation["agent"][0] * 5 + observation["agent"][1]  # current state
num_episodes = 100
initial_agent_pos = observation["agent"]
initial_target_pos = observation["target"]


def sample_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    return np.argmax(q_table[state])  # Exploit


print(env._get_obs())

for episode in range(num_episodes):
    obs, _ = env.reset()
    state = obs['agent'][0] * env.size + obs['agent'][1]
    done = False

    while not done:
        next_action = sample_action(state)
        observation, reward, terminated, truncated, info = env.step(
            next_action)
        next_state = observation["agent"][0] * 5 + observation["agent"][1]
        best_next_action = np.argmax(q_table[next_state])
        future_action = reward + gamma * q_table[next_state][best_next_action]
        q_table[state][next_action] += alpha * \
            (future_action - q_table[state][next_action])
        state = next_state
        done = terminated or truncated


print(env._get_obs())

for state in range(5 * 5):
    print(f"State {state}: {np.round(q_table[state], 2)}")


def plot_action_grid(q_table, grid_size, start_pos, end_pos):
    action_grid = np.argmax(q_table, axis=1).reshape(grid_size, grid_size)
    
    # Create a colormap for the actions
    cmap = plt.get_cmap('viridis', 4)
    
    # Plot the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(action_grid, cmap=cmap, origin='upper')
    
    # Add colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Down', 'Right', 'Up', 'Left'])
    
    # Add gridlines
    plt.grid(which='both', color='black', linestyle='-', linewidth=2)
    plt.xticks(np.arange(-.5, grid_size, 1), [])
    plt.yticks(np.arange(-.5, grid_size, 1), [])
    
    # Mark the starting and ending positions
    plt.scatter(start_pos[1], start_pos[0], color='red', s=200, marker='X', label='Start')
    plt.scatter(end_pos[1], end_pos[0], color='blue', s=200, marker='o', label='End')
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.title('Actions Taken in Each State')
    plt.savefig('action_grid.png')  # Save the plot as an image file
    plt.show()

# Plot the action grid with the initial starting and ending positions
plot_action_grid(q_table, 5, initial_agent_pos, initial_target_pos)

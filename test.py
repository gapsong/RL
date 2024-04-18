import gymnasium as gym

# Create the CartPole environment
env = gym.make('CartPole-v1')
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
# Initialize the environment and get the initial observation
state = env.reset()
done = False

# Track the total reward
total_reward = 0
action = 0
# Perform actions until the episode is over
while not done:
    # Render the environment to see the visualization
    env.render()

    # Select an action (0: push cart to the left, 1: push cart to the right)
    # Here, we are selecting actions randomly for demonstration purposes

    # Take the action and observe the new state, reward, and whether we're done
    state, reward, done, info, random = env.step(action)
    if state[2] < 0:
        action = 1
    else:
        action = 0

    print(random)
    # Update total reward
    total_reward += reward

# Print the total reward
print('Total Reward:', total_reward)

# Close the environment
env.close()

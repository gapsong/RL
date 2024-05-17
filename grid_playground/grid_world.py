import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


MAPS = {
    '8x8': np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    '5x5': np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])}


class GridWorldEnv(gym.Env):
    def __init__(self):
        self._level = MAPS['5x5']
        self.size = len(self._level[0])
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self._initialized = False

    def _get_obs(self):
        current_level = np.copy(self._level)
        current_level[self._agent_location[0]][self._agent_location[1]] = 2
        current_level[self._target_location[0]][self._target_location[1]] = 3
        return {"agent": self._agent_location, "target": self._target_location, "level": current_level}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None):
        # The positions are fixed, set them only once
        if not self._initialized:
            super().reset(seed=seed)
            self._agent_location = self._sample_location()
            self._target_location = self._sample_location(exclude=self._agent_location)
            self._initial_agent_location = self._agent_location.copy()
            self._initial_target_location = self._target_location.copy()
            self._initialized = True
        else:
            self._agent_location = self._initial_agent_location
            self._target_location = self._initial_target_location

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        temp_dir = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(
            temp_dir, self._target_location)

        if self._level[temp_dir[0]][temp_dir[1]] == 1:
            reward = 0
        else:
            if terminated:
                reward = 5
            else:
                reward = 0
            self._agent_location = temp_dir

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _sample_location(self, exclude=None):
        location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while self._level[location[0]][location[1]] != 0 or (exclude is not None and np.array_equal(location, exclude)):
            location = self.np_random.integers(0, self.size, size=2, dtype=int)
        return location

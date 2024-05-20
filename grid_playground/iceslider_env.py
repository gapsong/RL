import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IceSliderEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), player_start=(0, 0), goal=(4, 4), rocks=None, breaking_tiles=None, walls=None, sliding_tiles=None):
        super(IceSliderEnv, self).__init__()
        self.grid_size = grid_size
        # 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(grid_size[0], grid_size[1]))

        self.player_start = player_start
        self.goal = goal
        self.rocks = rocks if rocks else [(1, 1), (2, 3)]
        self.breaking_tiles = breaking_tiles if breaking_tiles else [
            (1, 2), (2, 2)]
        self.walls = walls if walls else [(0, 1), (1, 0)]
        self.sliding_tiles = sliding_tiles if sliding_tiles else [
            (1, 3), (3, 3)]
        self.reset()

    def reset(self):
        self.player_pos = list(self.player_start)
        self.grid = np.zeros(self.grid_size)
        for wall in self.walls:
            self.grid[wall] = 1  # 1 represents a wall
        for rock in self.rocks:
            self.grid[rock] = 5  # 5 represents a rock
        for tile in self.breaking_tiles:
            self.grid[tile] = 4  # 4 represents a breaking tile
        for slide in self.sliding_tiles:
            self.grid[slide] = 6  # 6 represents a sliding tile
        self.grid[self.goal] = 3  # 3 represents the goal
        self.grid[tuple(self.player_start)] = 2  # 2 represents the agent
        return self.grid.copy()

    def step(self, action):
        direction = self._get_direction(action)
        next_pos = [self.player_pos[0] + direction[0],
                    self.player_pos[1] + direction[1]]

        if not self._is_within_bounds(next_pos):
            return self.grid.copy(), -1, True, {}  # Out of bounds

        if self.grid[tuple(next_pos)] == 1:
            return self.grid.copy(), 0, False, {}  # Hit a wall

        if self.grid[tuple(next_pos)] == 5:  # Rock encountered
            rock_next_pos = [next_pos[0] + direction[0],
                             next_pos[1] + direction[1]]
            if self._is_within_bounds(rock_next_pos) and self.grid[tuple(rock_next_pos)] in [0, 4, 6]:
                self._move_rock(next_pos, rock_next_pos)
                self._move_agent(next_pos)
            else:
                return self.grid.copy(), 0, False, {}  # Cannot push the rock
        else:
            while self._is_within_bounds(next_pos) and self.grid[tuple(next_pos)] not in [1, 5]:
                if self.grid[tuple(next_pos)] == 0:  # Free tile
                    self._move_agent(next_pos)
                    break
                if self.grid[tuple(next_pos)] == 6:  # Sliding tile
                    self._move_agent(next_pos)
                    next_pos = [self.player_pos[0] + direction[0],
                                self.player_pos[1] + direction[1]]
                elif self.grid[tuple(next_pos)] == 4:  # Breaking tile
                    self.grid[tuple(next_pos)] = 7  # Mark as broken
                    self._move_agent(next_pos)
                    break

        if self.grid[tuple(self.player_pos)] == 0 and tuple(self.player_pos) != tuple(self.goal):
            return self.grid.copy(), -1, True, {}  # Stepped on broken tile

        if self.player_pos == list(self.goal):
            if not self._all_tiles_broken():
                return self.grid.copy(), -1, True, {}
            return self.grid.copy(), 1, True, {}

        return self.grid.copy(), 0, False, {}

    def _move_agent(self, next_pos):
        self.grid[tuple(self.player_pos)] = 0  # Clear previous agent position
        self.player_pos = next_pos
        self.grid[tuple(self.player_pos)] = 2  # Update agent position

    def _move_rock(self, rock_pos, new_pos):
        self.grid[tuple(rock_pos)] = 0  # Clear previous rock position
        self.grid[tuple(new_pos)] = 5  # Update rock position

    def _get_direction(self, action):
        if action == 0:
            return [-1, 0]  # Up
        elif action == 1:
            return [0, 1]  # Right
        elif action == 2:
            return [1, 0]  # Down
        elif action == 3:
            return [0, -1]  # Left
        return None

    def _is_within_bounds(self, pos):
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def _all_tiles_broken(self):
        return not np.any(self.grid == 4)

    def render(self, mode='human'):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == 0:
                    print(".", end=" ")
                elif self.grid[i, j] == 1:
                    print("1", end=" ")
                elif self.grid[i, j] == 2:
                    print("A", end=" ")
                elif self.grid[i, j] == 3:
                    print("T", end=" ")
                elif self.grid[i, j] == 4:
                    print("B", end=" ")
                elif self.grid[i, j] == 5:
                    print("R", end=" ")
                elif self.grid[i, j] == 6:
                    print("S", end=" ")
                elif self.grid[i, j] == 7:
                    print("X", end=" ")
            print()
        print()


if __name__ == "__main__":
    env = IceSliderEnv(
        grid_size=(5, 5),
        player_start=(0, 0),
        goal=(4, 4),
        rocks=[(1, 1), (2, 3)],
        breaking_tiles=[(1, 2), (2, 2)],
        walls=[(0, 0), (1, 0)],
        sliding_tiles=[(1, 3), (3, 3)]
    )
    env.reset()
    env.render()

    done = False
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        env.render()
        if done:
            if reward == 1:
                print("Reached the goal and broke all tiles!")
            else:
                print("Failed to break all tiles or stepped on a broken tile!")

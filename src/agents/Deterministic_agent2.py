import numpy as np
from agents.base_agent import BaseAgent
from numba import njit, cuda, float32
import time
from numba import cuda
from math import acos, pi
import math 
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider

ACTIONS = np.array([
    [0, 1], [1, 1], [1, 0], [1, -1],
    [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 0]
], dtype=np.int32)

class DeterministicAgent2(BaseAgent):
    """
    A deterministic agent that uses a Numba-powered A* planner to select actions,
    relying only on variables and observation, not env.
    """
    # Action mapping as a class variable
    ACTION_MAP = {
        (0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3,
        (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7, (0, 0): 8
    }
    ACTIONS = np.array([
        [0, 1], [1, 1], [1, 0], [1, -1],
        [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 0]
    ], dtype=np.int32)

    def __init__(self) -> None:
        """
        Initialize the DeterministicAgent2.
        """
        super().__init__()
        self.np_random = np.random.default_rng()
        self.grid_size = (32, 32)
        self.max_speed = 2.0
        self.wind_field = None
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1], dtype=np.float32)
        self.position_accumulator = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.position = np.zeros(2, dtype=np.float32)
        self.heuristic_table = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        self.emergency_actions = None

    def act_a_star(self, observation: np.ndarray) -> int:
        """
        Select an action using A* if possible, otherwise use greedy direction+efficiency.
        Args:
            observation (np.ndarray): The current observation from the environment.
        Returns:
            int: The action index to take.
        """
        self.heuristic_table = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        self.position = observation[:2]
        self.velocity = observation[2:4]
        wind_field_flat = observation[6:]
        self.wind_field = wind_field_flat.reshape(self.grid_size[0], self.grid_size[1], 2)

        # # # Precompute heuristic table if not already done
        # threadsperblock = (16, 16)
        # blockspergrid_x = int(np.ceil(self.grid_size[0] / threadsperblock[0]))
        # blockspergrid_y = int(np.ceil(self.grid_size[1] / threadsperblock[1]))

        # heuristic_table_device = cuda.to_device(self.heuristic_table)
        # wind_grid_device = cuda.to_device(self.wind_field)
        # goal_device = cuda.to_device(self.goal_position)
        # compute_heuristic_cuda[(blockspergrid_x, blockspergrid_y), threadsperblock](
        #     heuristic_table_device,
        #     wind_grid_device,
        #     goal_device,
        #     self.grid_size[0],
        #     self.grid_size[1]
        # )
        # self.heuristic_table = heuristic_table_device.copy_to_host()
        
        compute_recurrent_heuristic_njit(self.heuristic_table, self.wind_field, self.goal_position, self.grid_size)
        
        

        path,visit_counts,pos_visited = self.sailing_a_star_action(
            self.position,
            self.velocity,
            self.position_accumulator,
            self.wind_field,
            self.grid_size,
            self.goal_position,
            self.max_speed,
            max_iterations=15000,
            heuristic_table=self.heuristic_table if hasattr(self, 'heuristic_table') else None
        )

        
        if path is not None and isinstance(path,list) and len(path) > 1:
            direction = tuple(np.array(path[1]['direction']).astype(np.int32))
            action = self.ACTION_MAP.get(direction, 8)
            self.position = path[1]['pos']
            self.position_accumulator = path[1]['acc']
            self.velocity = path[1]['velocity']
            self.emergency_actions = path[2:]  # Store remaining actions for emergencies
            return action
        else:
            if path == 'MAX':
                # print("A* search with dist mode reached max iterations, falling back to Heuristic mode.")
                pass
            
            path,visit_counts,pos_visited = self.sailing_a_star_action(
            self.position,
            self.velocity,
            self.position_accumulator,
            self.wind_field,
            self.grid_size,
            self.goal_position,
            self.max_speed,
            mode='heuristic',
            max_iterations=15000,
            heuristic_table=self.heuristic_table if hasattr(self, 'heuristic_table') else None
        )
            # self.plot_a_star(visit_counts, pos_visited)
            # print(f"length of path: {len(path)}")
            if path is not None and isinstance(path,list) and len(path) > 1:
                direction = tuple(np.array(path[1]['direction']).astype(np.int32))
                action = self.ACTION_MAP.get(direction, 8)
                self.position = path[1]['pos']
                self.position_accumulator = path[1]['acc']
                self.velocity = path[1]['velocity']
                self.emergency_actions = path[2:]
                return action
            else:
                if path == 'MAX':
                    # print("A* search with heursitic mode reached max iterations, falling back to greedy mode.")
                    pass
                if self.emergency_actions is not None and len(self.emergency_actions) > 0:
                    direction = tuple(np.array(self.emergency_actions[0]['direction']).astype(np.int32))
                    action = self.ACTION_MAP.get(direction, 8)
                    self.position = self.emergency_actions[0]['pos']
                    self.position_accumulator = self.emergency_actions[0]['acc']
                    self.velocity = self.emergency_actions[0]['velocity']
                    self.emergency_actions = self.emergency_actions[1:]
                    return action
                # print("A* failed, using greedy fallback.")
                
                return self._greedy_fallback(observation)

    def _greedy_fallback(self, observation: np.ndarray) -> int:
        """
        Fallback action selection using greedy direction and sailing efficiency.
        Args:
            observation (np.ndarray): The current observation from the environment.
        Returns:
            int: The action index to take.
        """
        # print("A* failed, using greedy fallback.")
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        current_pos = np.array([x, y])
        wind_vec = np.array([wx, wy])
        wind_norm = np.linalg.norm(wind_vec)
        wind_dir = wind_vec / wind_norm if wind_norm > 1e-8 else np.array([0.0, 0.0])

        direction_to_goal = self.goal_position - current_pos
        direction_to_goal_norm = np.linalg.norm(direction_to_goal)
        direction_to_goal = direction_to_goal / direction_to_goal_norm if direction_to_goal_norm > 1e-8 else np.array([0.0, 0.0])

        scores = []
        for d in self.ACTIONS:
            d_norm = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-8 else d
            similarity = np.dot(d_norm, direction_to_goal)
            efficiency = self.calculate_sailing_efficiency(d_norm, wind_dir)
            score = 0.7 * similarity + 0.3 * efficiency
            scores.append(score)
        best_idx = np.argmax(scores)
        best_direction = tuple(self.ACTIONS[best_idx])
        action = self.ACTION_MAP[best_direction]
        return action

    def reset(self) -> None:
        """
        Reset the agent's internal state if needed.
        """
        pass

    def seed(self, seed: int = None) -> None:
        """
        Seed the agent's random number generator.
        Args:
            seed (int, optional): The seed value.
        """
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        Args:
            path (str): The file path to save to.
        """
        pass

    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        Args:
            path (str): The file path to load from.
        """
        pass

    def act(self, observation: np.ndarray) -> int:
        """
        Use the precomputed value iteration policy to select an action.
        Args:
            observation (np.ndarray): The current observation from the environment.
        Returns:
            int: The action index to take.
        """
        return self.act_a_star(observation)
    
    def plot_a_star(self,visit_counts: np.ndarray, pos_visited: List[Tuple[int, int]]) -> None:
        """
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Heuristic Table Heatmap
        im0 = axes[0].imshow(self.heuristic_table.T, cmap='hot', interpolation='nearest')
        axes[0].set_title("Heuristic Table Heatmap")
        axes[0].set_xlabel("X Position")
        axes[0].set_ylabel("Y Position")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Visit Counts Heatmap
        im1 = axes[1].imshow(visit_counts.T, origin='lower', cmap='hot')
        axes[1].set_title("A* Visited Positions Heatmap")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        
        
        # Example data
        # pos_visited = [(i, i) for i in range(10)]
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.set_title("Mist of Visited Positions (Slider)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim(-0.5, visit_counts.shape[0] - 0.5)
        ax2.set_ylim(-0.5, visit_counts.shape[1] - 0.5)
        ax2.invert_yaxis()

        # Slider setup
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        slider = Slider(ax_slider, 'Step', 0, len(pos_visited)-1, valinit=0, valstep=1)

        scatter_artists = []

        def on_slider(val):
            frame = int(val)
            # Remove previous scatter artists
            for artist in scatter_artists:
                artist.remove()
            scatter_artists.clear()
            # Show previous position (if any)
            if frame > 0:
                x_prev, y_prev = pos_visited[frame-1]
                prev_artist = ax2.scatter(x_prev, y_prev, c='blue', alpha=0.3, s=120, label='Previous')
                scatter_artists.append(prev_artist)
            # Show current position
            x, y = pos_visited[frame]
            curr_artist = ax2.scatter(x, y, c='red', alpha=1.0, s=120, label='Current')
            scatter_artists.append(curr_artist)
            ax2.legend(loc='upper right')
            fig2.canvas.draw_idle()

        slider.on_changed(on_slider)
        on_slider(0)  # Initialize

        plt.show()
        plt.close(fig2)
        plt.close(fig)


    @staticmethod
    def sailing_a_star_action(
        start: np.ndarray,
        velocity: np.ndarray,
        acc: np.ndarray,
        wind_grid: np.ndarray,
        grid_size: tuple,
        goal: np.ndarray,
        max_speed: float,
        mode: str = 'dist',
        inertia_factor: float = 0.3,
        boat_performance: float = 0.4,
        max_iterations: int = None,
        heuristic_table: np.ndarray = None
    ) -> list:
        """
        Perform A* search to find a path to the goal.
        Args:
            start (np.ndarray): Starting position.
            velocity (np.ndarray): Starting velocity.
            acc (np.ndarray): Starting accumulator.
            wind_grid (np.ndarray): Wind field grid.
            grid_size (tuple): Size of the grid.
            goal (np.ndarray): Goal position.
            max_speed (float): Maximum speed.
            inertia_factor (float): Inertia factor.
            boat_performance (float): Boat performance factor.
            max_iterations (int, optional): Maximum iterations for A*.
            heuristic_table (np.ndarray, optional): Precomputed heuristic table.
        Returns:
            list: Path as a list of state dictionaries, or None if not found.
        """
        def heuristic(pos: tuple) -> float:
            x, y = int(pos[0]), int(pos[1])
            if heuristic_table is not None:
                x = np.clip(x, 0, grid_size[0] - 1)
                y = np.clip(y, 0, grid_size[1] - 1)
                return heuristic_table[x, y]
            else:
                raise ValueError("heuristic_table is None, cannot compute heuristic.")

        import heapq
        open_set = []
        start_tuple = (int(start[0]), int(start[1]))
        start_velocity = np.array(velocity, dtype=np.float32)
        start_acc = np.array(acc, dtype=np.float32)
        start_wind = wind_grid[start_tuple[1], start_tuple[0]]

        counter = 0  # Unique tie-breaker

        heapq.heappush(open_set, (
            # np.linalg.norm(start_tuple - goal),
            np.linalg.norm(start_tuple - goal)**1.2 if mode=='dist' else heuristic(start_tuple),
            0,
            counter,
            start_tuple,
            tuple(start_velocity.tolist()),
            tuple(start_acc.tolist()),
            [{
                'pos': start_tuple,
                'velocity': tuple(start_velocity.tolist()),
                'direction': (0.0, 0.0),
                'wind': tuple(start_wind.tolist()),
                'efficiency': 0.0
            }]
        ))
        visit_counts = np.zeros(grid_size, dtype=np.int32)
        visit_counts[tuple(start_tuple)] = 1
        pos_visited = list()
        pos_visited.append(tuple(start_tuple))
        # pos_with_highest_y = np.array([0, 0], dtype=np.int32)
        # highest_y = 0
        visited = set()
        iterations = 0
        if isinstance(grid_size, int):
            grid_size_arr = np.array([grid_size, grid_size], dtype=np.int32)
        
        else:
            grid_size_arr = np.array(grid_size, dtype=np.int32)

        while open_set:
            if max_iterations is not None and iterations >= max_iterations:
                return 'MAX',visit_counts,pos_visited
            iterations += 1

            est_total, cost_so_far, current_counter, current, velocity, acc, path = heapq.heappop(open_set)
            # if current[1] > highest_y:
            #     highest_y = current[1]
            #     pos_with_highest_y = np.array(current, dtype=np.int32)

            velocity = np.array(velocity, dtype=np.float32)
            acc = np.array(acc, dtype=np.float32)

            if np.linalg.norm(np.array(current) - np.array(goal)) < 1.0:
                return path,visit_counts,pos_visited
            visit_counts[tuple(current)] += 1
            pos_visited.append(tuple(current))
            if visit_counts[tuple(current)] > 10:
                # heuristic_table[tuple(current)] *= 2
                continue
            
            neighbors = get_neighbors_numba(
                np.array(current, dtype=np.float32),
                velocity.astype(np.float32),
                acc.astype(np.float32),
                wind_grid,
                grid_size_arr,
                boat_performance,
                max_speed,
                inertia_factor
            )
            
            
            visited.add((current, tuple(np.round(velocity, 1)), tuple(np.round(acc, 1))))
            counter= current_counter+1
            for i in range(neighbors.shape[0]):
                neighbor = tuple(neighbors[i, 0:2].astype(np.int32))
                nvel = tuple(neighbors[i, 2:4])
                nacc = tuple(neighbors[i, 4:6])
                direction = tuple(neighbors[i, 6:8])
                wind_direction = tuple(neighbors[i, 8:10])
                sailing_efficiency = float(neighbors[i, 10])
                state_id = (neighbor, tuple(np.round(nvel, 2)), tuple(np.round(nacc, 2)))
                if state_id in visited or visit_counts[tuple(neighbor)] > 10:
                    # heuristic_table[neighbor[0], neighbor[1]] *=2
                    continue

                
                heapq.heappush(open_set, (
                    # np.linalg.norm(np.array(neighbor) - np.array(goal)),
                    0.8*(cost_so_far+1)+np.linalg.norm(np.array(neighbor) - np.array(goal))**1.2+ np.random.uniform(-0.1,0.1) if mode=='dist' else 0.8*(cost_so_far+1)+heuristic(neighbor)+ np.random.uniform(-0.1,0.1),
                    cost_so_far+1,
                    counter,
                    neighbor,
                    nvel,
                    nacc,
                    path + [{
                        'pos': neighbor,
                        'velocity': nvel,
                        'acc': nacc,
                        'direction': direction,
                        'wind': wind_direction,
                        'efficiency': sailing_efficiency
                    }]
                ))
        return None,visit_counts,pos_visited

    def calculate_sailing_efficiency(self, boat_direction: np.ndarray, wind_direction: np.ndarray) -> float:
        """
        Calculate sailing efficiency based on the angle between boat direction and wind.
        Args:
            boat_direction (np.ndarray): Normalized vector of boat's desired direction.
            wind_direction (np.ndarray): Normalized vector of wind direction (where wind is going TO).
        Returns:
            float: Sailing efficiency between 0.05 and 1.0.
        """
        wind_from = -wind_direction
        wind_angle = np.arccos(np.clip(np.dot(wind_from, boat_direction), -1.0, 1.0))
        if wind_angle < np.pi / 4:
            sailing_efficiency = 0.05
        elif wind_angle < np.pi / 2:
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi / 4) / (np.pi / 4)
        elif wind_angle < 3 * np.pi / 4:
            sailing_efficiency = 1.0
        else:
            sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3 * np.pi / 4) / (np.pi / 4)
            sailing_efficiency = max(0.5, sailing_efficiency)
        return sailing_efficiency

@njit
def calculate_new_velocity_numba(current_velocity, wind, direction, boat_performance, max_speed, inertia_factor):
    wind_norm = np.sqrt(wind[0]**2 + wind[1]**2)
    new_velocity = np.zeros(2, dtype=np.float32)
    wind_normalized = np.array([0.0, 0.0], dtype=np.float32)
    sailing_efficiency = 0.0

    if wind_norm > 0.0:
        wind_normalized = wind / wind_norm

        direction_norm = np.sqrt(direction[0]**2 + direction[1]**2)
        if direction_norm < 1e-10:
            direction_normalized = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction_normalized = direction / direction_norm

        dot = direction_normalized[0] * -wind_normalized[0] + direction_normalized[1] * -wind_normalized[1]
        angle = np.arccos(min(1.0, max(-1.0, dot)))

        if angle < np.pi / 4:
            sailing_efficiency = 0.05
        elif angle < np.pi / 2:
            sailing_efficiency = 0.5 + 0.5 * (angle - np.pi / 4) / (np.pi / 4)
        elif angle < 3 * np.pi / 4:
            sailing_efficiency = 1.0
        else:
            sailing_efficiency = max(0.5, 1.0 - 0.5 * (angle - 3 * np.pi / 4) / (np.pi / 4))

        theoretical_velocity = direction * sailing_efficiency * wind_norm * boat_performance
        speed = np.sqrt(theoretical_velocity[0]**2 + theoretical_velocity[1]**2)
        if speed > max_speed:
            theoretical_velocity = theoretical_velocity / speed * max_speed

        new_velocity = theoretical_velocity + inertia_factor * (current_velocity - theoretical_velocity)
        speed = np.sqrt(new_velocity[0]**2 + new_velocity[1]**2)
        if speed > max_speed:
            new_velocity = new_velocity / speed * max_speed
    else:
        new_velocity = inertia_factor * current_velocity

    return new_velocity.astype(np.float32), wind_normalized.astype(np.float32), sailing_efficiency

@njit
def get_neighbors_numba(pos, velocity, acc, wind_grid, grid_size, boat_performance, max_speed, inertia_factor):
    actions = np.array([
        [0, 1], [1, 1], [1, 0], [1, -1],
        [0, -1], [-1, -1], [-1, 0], [-1, 1], [0,0]
    ], dtype=np.int32)

    n_actions = actions.shape[0]
    neighbors = np.zeros((n_actions, 11), dtype=np.float32)

    for i in range(n_actions):
        direction = np.array([float(actions[i, 0]), float(actions[i, 1])], dtype=np.float32)
        x = min(max(int(pos[0]), 0), grid_size[0]-1)
        y = min(max(int(pos[1]), 0), grid_size[1]-1)

        wind = wind_grid[y, x]
        new_velocity, wind_direction, sailing_efficiency = calculate_new_velocity_numba(
            velocity, wind, direction, boat_performance, max_speed, inertia_factor
        )
        new_acc = acc + new_velocity
        new_position_float = pos + new_acc
        new_position = np.round(new_position_float).astype(np.int32)
        new_acc2 = new_position_float - new_position.astype(np.float32)
        new_position = np.minimum(np.maximum(new_position, np.array([0, 0], dtype=np.int32)), grid_size - 1)

        neighbors[i, 0:2] = new_position.astype(np.float32)
        neighbors[i, 2:4] = new_velocity
        neighbors[i, 4:6] = new_acc2
        neighbors[i, 6:8] = direction
        neighbors[i, 8:10] = wind_direction
        neighbors[i, 10] = sailing_efficiency

    return neighbors

@cuda.jit
def compute_heuristic_cuda(heuristic_table, wind_grid, goal, grid_x, grid_y):
    x, y = cuda.grid(2)
    if x < grid_x and y < grid_y:
        pos0 = float(x)
        pos1 = float(y)
        goal0 = float(goal[0])
        goal1 = float(goal[1])
        dx = goal0 - pos0
        dy = goal1 - pos1
        dist = math.sqrt(dx * dx + dy * dy)**1.15

        # Wind at this cell
        wind_x = wind_grid[x, y, 0]
        wind_y = wind_grid[x, y, 1]
        wind_norm = math.sqrt(wind_x * wind_x + wind_y * wind_y)
        if wind_norm > 1e-6:
            wind_dir_x = wind_x / wind_norm
            wind_dir_y = wind_y / wind_norm
        else:
            wind_dir_x = 0.0
            wind_dir_y = 0.0

        # Direction to goal (normalized)
        to_goal_norm = dist
        if to_goal_norm > 1e-6:
            to_goal_dir_x = dx / to_goal_norm
            to_goal_dir_y = dy / to_goal_norm
        else:
            to_goal_dir_x = 0.0
            to_goal_dir_y = 0.0

        # Calculate sailing efficiency (same logic as in agent)
        wind_from_x = -wind_dir_x
        wind_from_y = -wind_dir_y
        dot = wind_from_x * to_goal_dir_x + wind_from_y * to_goal_dir_y
        dot = min(1.0, max(-1.0, dot))  # Clamp dot to [-1, 1]
        wind_angle = math.acos(dot)
        if wind_angle < math.pi / 4:
            sailing_efficiency = 0.05
        elif wind_angle < math.pi / 2:
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - math.pi / 4) / (math.pi / 4)
        elif wind_angle < 3 * math.pi / 4:
            sailing_efficiency = 1.0
        else:
            sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3 * math.pi / 4) / (math.pi / 4)

        # Penalize heuristic by efficiency (lower efficiency = higher cost)
        heuristic = dist / max(sailing_efficiency, 0.1)
        heuristic_table[x, y] = heuristic
        

@njit
def compute_recurrent_heuristic_njit(heuristic_table, wind_grid, goal, grid_size, boat_performance=0.4, max_speed=2.0, inertia_factor=0.3):
    X, Y = grid_size
    heuristic_table[:] = np.inf
    gx, gy = int(goal[0]), int(goal[1])
    heuristic_table[gx, gy] = 0.0

    # 8-connected grid
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    # Use a simple queue for wavefront expansion (Dijkstra's with uniform cost)
    queue = [(gx, gy)]
    while len(queue) > 0:
        x, y = queue.pop(0)
        cost = heuristic_table[x, y]
        wind = wind_grid[x, y]
        wind_norm = np.sqrt(wind[0]**2 + wind[1]**2)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < X and 0 <= ny < Y:
                direction = np.array([dx, dy], dtype=np.float32)
                direction_norm = np.sqrt(direction[0]**2 + direction[1]**2)
                if direction_norm < 1e-8:
                    continue
                direction_normalized = direction / direction_norm

                # Sailing efficiency (same as in env)
                if wind_norm > 1e-8:
                    wind_normalized = wind / wind_norm
                    dot = wind_normalized[0]*direction_normalized[0] + wind_normalized[1]*direction_normalized[1]
                    if dot < -1.0:
                        dot = -1.0
                    elif dot > 1.0:
                        dot = 1.0
                    wind_angle = np.arccos(dot)
                    if wind_angle < np.pi / 4:
                        sailing_efficiency = 0.05
                    elif wind_angle < np.pi / 2:
                        sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi / 4) / (np.pi / 4)
                    elif wind_angle < 3 * np.pi / 4:
                        sailing_efficiency = 1.0
                    else:
                        sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3 * np.pi / 4) / (np.pi / 4)
                else:
                    sailing_efficiency = 0.0

                # Theoretical velocity
                theoretical_velocity = direction * sailing_efficiency * wind_norm * boat_performance
                speed = np.sqrt(theoretical_velocity[0]**2 + theoretical_velocity[1]**2)
                if speed > max_speed:
                    theoretical_velocity = theoretical_velocity / speed * max_speed
                    speed = max_speed
                    
                # For heuristic, use the time to traverse the step (distance / speed)
                step_dist = 1
                if speed > 1e-4:
                    transition_cost = step_dist / (speed)**4 
                else:
                    transition_cost = 1e6  # Large penalty if can't move

                # Optionally, add distance to goal for tie-breaking
                dist_to_goal = np.sqrt((nx - gx)**2 + (ny - gy)**2)
                total_cost = cost + transition_cost 

                if total_cost < heuristic_table[nx, ny]:
                    heuristic_table[nx, ny] = total_cost
                    queue.append((nx, ny))
                    
                          
        
import math

@cuda.jit
def get_neighbors_cuda(pos, velocity, acc, wind_grid, grid_size, boat_performance, max_speed, inertia_factor, neighbors, actions):
    i = cuda.threadIdx.x
    n_actions = actions.shape[0]
    if i < n_actions:
        direction = cuda.local.array(2, float32)
        direction[0] = float(actions[i, 0])
        direction[1] = float(actions[i, 1])

        x = min(max(int(pos[0]), 0), grid_size[0]-1)
        y = min(max(int(pos[1]), 0), grid_size[1]-1)

        wind_x = wind_grid[y, x, 0]
        wind_y = wind_grid[y, x, 1]
        wind_norm = math.sqrt(wind_x * wind_x + wind_y * wind_y)
        wind_normalized_x = 0.0
        wind_normalized_y = 0.0
        sailing_efficiency = 0.0

        if wind_norm > 1e-6:
            wind_normalized_x = wind_x / wind_norm
            wind_normalized_y = wind_y / wind_norm

            direction_norm = math.sqrt(direction[0]**2 + direction[1]**2)
            if direction_norm < 1e-10:
                direction_normalized_x = 1.0
                direction_normalized_y = 0.0
            else:
                direction_normalized_x = direction[0] / direction_norm
                direction_normalized_y = direction[1] / direction_norm

            dot = direction_normalized_x * -wind_normalized_x + direction_normalized_y * -wind_normalized_y
            dot = min(1.0, max(-1.0, dot))
            angle = math.acos(dot)

            if angle < math.pi / 4:
                sailing_efficiency = 0.05
            elif angle < math.pi / 2:
                sailing_efficiency = 0.5 + 0.5 * (angle - math.pi / 4) / (math.pi / 4)
            elif angle < 3 * math.pi / 4:
                sailing_efficiency = 1.0
            else:
                sailing_efficiency = max(0.5, 1.0 - 0.5 * (angle - 3 * math.pi / 4) / (math.pi / 4))

            theoretical_velocity_x = direction[0] * sailing_efficiency * wind_norm * boat_performance
            theoretical_velocity_y = direction[1] * sailing_efficiency * wind_norm * boat_performance
            speed = math.sqrt(theoretical_velocity_x**2 + theoretical_velocity_y**2)
            if speed > max_speed:
                theoretical_velocity_x = theoretical_velocity_x / speed * max_speed
                theoretical_velocity_y = theoretical_velocity_y / speed * max_speed

            new_velocity_x = theoretical_velocity_x + inertia_factor * (velocity[0] - theoretical_velocity_x)
            new_velocity_y = theoretical_velocity_y + inertia_factor * (velocity[1] - theoretical_velocity_y)
            speed = math.sqrt(new_velocity_x**2 + new_velocity_y**2)
            if speed > max_speed:
                new_velocity_x = new_velocity_x / speed * max_speed
                new_velocity_y = new_velocity_y / speed * max_speed
        else:
            new_velocity_x = inertia_factor * velocity[0]
            new_velocity_y = inertia_factor * velocity[1]

        new_acc_x = acc[0] + new_velocity_x
        new_acc_y = acc[1] + new_velocity_y
        new_position_float_x = pos[0] + new_acc_x
        new_position_float_y = pos[1] + new_acc_y
        new_position_x = int(round(new_position_float_x))
        new_position_y = int(round(new_position_float_y))
        new_acc2_x = new_position_float_x - float(new_position_x)
        new_acc2_y = new_position_float_y - float(new_position_y)
        new_position_x = min(max(new_position_x, 0), grid_size[0] - 1)
        new_position_y = min(max(new_position_y, 0), grid_size[1] - 1)

        neighbors[i, 0] = float(new_position_x)
        neighbors[i, 1] = float(new_position_y)
        neighbors[i, 2] = new_velocity_x
        neighbors[i, 3] = new_velocity_y
        neighbors[i, 4] = new_acc2_x
        neighbors[i, 5] = new_acc2_y
        neighbors[i, 6] = direction[0]
        neighbors[i, 7] = direction[1]
        neighbors[i, 8] = wind_normalized_x
        neighbors[i, 9] = wind_normalized_y
        neighbors[i, 10] = sailing_efficiency
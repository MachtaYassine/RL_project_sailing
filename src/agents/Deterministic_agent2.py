import numpy as np
from agents.base_agent import BaseAgent
from numba import njit
import time

class DeterministicAgent2(BaseAgent):
    """
    A deterministic agent that uses a Numba-powered A* planner to select actions,
    relying only on variables and observation, not env.
    """

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        # You must set these externally before calling act:
        self.grid_size = (32,32)
        self.max_speed = 2.0
        self.wind_field = None
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1], dtype=np.float32)  # Example goal position
        self.position_accumulator = np.zeros(2, dtype=np.float32)  # Example position accumulator
        self.velocity = np.zeros(2, dtype=np.float32)  # Example velocity accumulator
        self.position= np.zeros(2, dtype=np.float32)  # Example position
        self.heuristic_table = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)

    def act_a_star(self, observation: np.ndarray) -> int:
        """
        Select an action using A* if possible, otherwise use greedy direction+efficiency.
        """
        # Try A* (you must set up the required attributes externally)
        
        self.position = observation[:2]
        self.velocity = observation[2:4]
        wind_field_flat= observation[6:]
        self.wind_field = wind_field_flat.reshape(self.grid_size[0], self.grid_size[1], 2)
        # print(f" Observed position and velocity: {self.position} {self.velocity}")
        # print(f'==================')
        # precompute heuristic table if not already done
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(self.grid_size[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(self.grid_size[1] / threadsperblock[1]))

        heuristic_table_device = cuda.to_device(self.heuristic_table)
        wind_grid_device = cuda.to_device(self.wind_field)
        goal_device = cuda.to_device(self.goal_position)
        start_time = time.time()
        compute_heuristic_cuda[(blockspergrid_x, blockspergrid_y), threadsperblock](
            heuristic_table_device,
            wind_grid_device,
            goal_device,
            self.grid_size[0],
            self.grid_size[1]
        )
        self.heuristic_table = heuristic_table_device.copy_to_host()
        end_time = time.time()
        # print(f" Heuristic table computed in {end_time - start_time:.4f} seconds")
        #check the heuristic table was changed
        # print(f" heuristic table: {self.heuristic_table}")
        
        start_time = time.time()
        path = self.sailing_a_star_action(
            self.position,
            self.velocity,
            self.position_accumulator,  # or your position_accumulator if you track it
            self.wind_field,
            self.grid_size,
            self.goal_position,
            self.max_speed,
            max_iterations=10000,
            heuristic_table=self.heuristic_table if hasattr(self, 'heuristic_table') else None
        )
        end_time = time.time()
        # print(f" A* pathfinding took {end_time - start_time:.4f} seconds")
        # print(f"path is None: {path is None}")
        # print(f"passed A* ")
        

        action_map = {
            (0,1): 0, (1,1): 1, (1,0): 2, (1,-1): 3,
            (0,-1): 4, (-1,-1): 5, (-1,0): 6, (-1,1): 7, (0,0): 8
        }

        if path is not None and len(path) > 1:
            direction = tuple(path[1]['direction'])
            direction=(direction[0].astype(np.int32), direction[1].astype(np.int32))
            action = action_map.get(direction, 8)
            self.position = path[1]['pos']
            self.position_accumulator = path[1]['acc']
            self.velocity = path[1]['velocity']
            # print(f"predicted position and velocity: {self.position} {self.velocity}")
            return action
        else:
            if path is None:
                print(f" A* path is None")
            else:
                print(f" A* path is too short: {len(path)}")

        # print(f" Passed path assignment")
        # --- Fallback: Greedy direction+efficiency logic ---
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        current_pos = np.array([x, y])
        wind_vec = np.array([wx, wy])
        wind_norm = np.linalg.norm(wind_vec)
        if wind_norm > 1e-8:
            wind_dir = wind_vec / wind_norm
        else:
            wind_dir = np.array([0.0, 0.0])

        direction_to_goal = self.goal_position - current_pos
        direction_to_goal_norm = np.linalg.norm(direction_to_goal)
        if direction_to_goal_norm > 1e-8:
            direction_to_goal = direction_to_goal / direction_to_goal_norm
        else:
            direction_to_goal = np.array([0.0, 0.0])

        directions = np.array([
            [0, 1], [1, 1], [1, 0], [1, -1],
            [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 0]
        ])
        # print(f' starting scores')
        scores = []
        for d in directions:
            d_norm = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-8 else d
            similarity = np.dot(d_norm, direction_to_goal)
            efficiency = self.calculate_sailing_efficiency(d_norm, wind_dir)
            score = 0.7 * similarity + 0.3 * efficiency
            scores.append(score)
        # print(f' finished scores')
        best_idx = np.argmax(scores)
        best_direction = tuple(directions[best_idx])
        action = action_map[best_direction]
        # print(f"action is {action}")
        return action

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    

    def act(self, observation: np.ndarray) -> int:
        """
        Use the precomputed value iteration policy to select an action.
        """
        
        return self.act_a_star(observation)

    def precompute_heuristic_table(self):
        """
        Precompute the heuristic for all grid positions and store in self.heuristic_table.
        Assumes wind_field and goal_position are set.
        """
        grid_x, grid_y = self.grid_size
        self.heuristic_table = np.zeros((grid_x, grid_y), dtype=np.float32)
        goal = self.goal_position
        wind_grid = self.wind_field
        for x in range(grid_x):
            for y in range(grid_y):
                pos = np.array([x, y], dtype=np.float32)
                dist = (np.linalg.norm(pos - goal)) ** 1.2
                wind = wind_grid[y, x]
                wind_norm = np.linalg.norm(wind)
                direction_to_goal = goal - pos
                if np.linalg.norm(direction_to_goal) > 1e-8:
                    direction_to_goal /= np.linalg.norm(direction_to_goal)
                else:
                    direction_to_goal = np.zeros(2)
                wind_dir = wind / wind_norm if wind_norm > 1e-8 else np.zeros(2)
                wind_from = -wind_dir
                angle = np.arccos(np.clip(np.dot(wind_from, direction_to_goal), -1.0, 1.0))
                penalty = 1.0
                if angle < np.pi/4:
                    penalty = 2.0
                elif angle < np.pi/2:
                    penalty = 1.5 + 0.5 * (angle - np.pi/4) / (np.pi/4)
                elif angle < 3*np.pi/4:
                    penalty = 1.0
                else:
                    penalty = 1.0 - 0.5 * (angle - 3*np.pi/4) / (np.pi/4)
                    penalty = max(0.5, penalty)
                self.heuristic_table[x, y] = dist * penalty

    @staticmethod
    def sailing_a_star_action(
        start, velocity, acc, wind_grid, grid_size, goal, max_speed,
        inertia_factor=0.3, boat_performance=0.4, max_iterations=None,
        heuristic_table=None
    ):
        def heuristic(pos):
            x, y = int(pos[0]), int(pos[1])
            if heuristic_table is not None:
                x = np.clip(x, 0, grid_size[0]-1)
                y = np.clip(y, 0, grid_size[1]-1)
                return heuristic_table[x, y]
            else:
                print(f" Warning: heuristic_table is None, using default heuristic")
                import sys
                sys.exit(0)
        
        import heapq
        open_set = []
        start_tuple = (int(start[0]), int(start[1]))
        start_velocity = np.array(velocity, dtype=np.float32)
        start_acc = np.array(acc, dtype=np.float32)
        start_wind = wind_grid[start_tuple[1], start_tuple[0]]

        counter = 0  # Unique tie-breaker

        heapq.heappush(open_set, (
            heuristic(start_tuple),
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
        
        # print(f" Initial parameters: {start_tuple} {start_velocity} {start_acc} {start_wind}")

        pos_with_highest_y = np.array([0, 0], dtype=np.int32)
        highest_y=0
        visited = set()
        iterations = 0
        # Ensure grid_size is always a tuple/list/array of length 2
        if isinstance(grid_size, int):
            grid_size_arr = np.array([grid_size, grid_size], dtype=np.int32)
        else:
            grid_size_arr = np.array(grid_size, dtype=np.int32)

        while open_set:
            if max_iterations is not None and iterations >= max_iterations:
                print(f" A* iterations exceeded: {iterations}")
                return None
            iterations += 1

            est_total, cost_so_far, _, current, velocity, acc, path = heapq.heappop(open_set)
            if current[1] > highest_y:
                highest_y = current[1]
                pos_with_highest_y = np.array(current, dtype=np.int32)
                # print(f" A* new best position: {pos_with_highest_y} {highest_y}")

            velocity = np.array(velocity, dtype=np.float32)
            acc = np.array(acc, dtype=np.float32)

            if np.linalg.norm(np.array(current) - np.array(goal)) < 1.0:
                return path

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
            visited.add((current, tuple(np.round(velocity, 2)), tuple(np.round(acc, 2))))
            
            
            for i in range(neighbors.shape[0]):
                
                neighbor = tuple(neighbors[i, 0:2].astype(np.int32))
                nvel = tuple(neighbors[i, 2:4])
                nacc = tuple(neighbors[i, 4:6])
                direction = tuple(neighbors[i, 6:8])
                wind_direction = tuple(neighbors[i, 8:10])
                sailing_efficiency = float(neighbors[i, 10])
                state_id = (neighbor, tuple(np.round(nvel, 2)), tuple(np.round(nacc, 2)))

                if state_id in visited:
                    continue

                counter += 1
                heapq.heappush(open_set, (
                    cost_so_far + 1.0 + heuristic(neighbor),
                    cost_so_far + 1.0,
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
        print(f" A* no path found, returning best position: {pos_with_highest_y}")
        return None
    
    def calculate_sailing_efficiency(self,boat_direction, wind_direction):
        """
        Calculate sailing efficiency based on the angle between boat direction and wind.
        
        Args:
            boat_direction: Normalized vector of boat's desired direction
            wind_direction: Normalized vector of wind direction (where wind is going TO)
            
        Returns:
            sailing_efficiency: Float between 0.05 and 1.0 representing how efficiently the boat can sail
        """
        # Invert wind direction to get where wind is coming FROM
        wind_from = -wind_direction
        
        # Calculate angle between wind and direction
        wind_angle = np.arccos(np.clip(
            np.dot(wind_from, boat_direction), -1.0, 1.0))
        
        # Calculate sailing efficiency based on angle to wind
        if wind_angle < np.pi/4:  # Less than 45 degrees to wind
            sailing_efficiency = 0.05  # Small but non-zero efficiency in no-go zone
        elif wind_angle < np.pi/2:  # Between 45 and 90 degrees
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)  # Linear increase to 1.0
        elif wind_angle < 3*np.pi/4:  # Between 90 and 135 degrees
            sailing_efficiency = 1.0  # Maximum efficiency
        else:  # More than 135 degrees
            sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4)  # Linear decrease
            sailing_efficiency = max(0.5, sailing_efficiency)  # But still decent
        
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

from numba import cuda
from math import acos, pi
@cuda.jit
def compute_heuristic_cuda(heuristic_table, wind_grid, goal, grid_x, grid_y):
    x, y = cuda.grid(2)
    if x < grid_x and y < grid_y:
        pos0 = float(x)
        pos1 = float(y)
        goal0 = float(goal[0])
        goal1 = float(goal[1])
        dist = ((pos0 - goal0)**2 + (pos1 - goal1)**2) ** 1.2  # **1.2
        wind0 = wind_grid[y, x, 0]
        wind1 = wind_grid[y, x, 1]
        wind_norm = (wind0**2 + wind1**2) ** 0.5
        dir0 = goal0 - pos0
        dir1 = goal1 - pos1
        dir_norm = (dir0**2 + dir1**2) ** 0.5
        if dir_norm > 1e-8:
            dir0 /= dir_norm
            dir1 /= dir_norm
        else:
            dir0 = 0.0
            dir1 = 0.0
        if wind_norm > 1e-8:
            wind_dir0 = wind0 / wind_norm
            wind_dir1 = wind1 / wind_norm
        else:
            wind_dir0 = 0.0
            wind_dir1 = 0.0
        wind_from0 = -wind_dir0
        wind_from1 = -wind_dir1
        dot = wind_from0 * dir0 + wind_from1 * dir1
        dot = min(1.0, max(-1.0, dot))
       
        angle = acos(dot)
        penalty = 1.0
        if angle < pi/4:
            penalty = 2.0
        elif angle < pi/2:
            penalty = 1.5 + 0.5 * (angle - pi/4) / (pi/4)
        elif angle < 3*pi/4:
            penalty = 1.0
        else:
            penalty = 1.0 - 0.5 * (angle - 3*pi/4) / (pi/4)
            penalty = max(0.5, penalty)
        heuristic_table[x, y] = dist 
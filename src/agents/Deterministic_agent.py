"""
Naive Agent for the Sailing Challenge

This file provides a simple agent that always goes north.
Students can use it as a reference for how to implement a sailing agent.
"""

import numpy as np # type: ignore
from agents.base_agent import BaseAgent

class DeterministicAgent(BaseAgent):
    """
    A naive agent for the Sailing Challenge.
    
    This is a very simple agent that always chooses to go North,
    regardless of wind conditions or position. It serves as a minimal
    working example that students can build upon.
    """
    
    def __init__(self):
        """Initialize the agent."""
        super().__init__()
        self.np_random = np.random.default_rng()
        self.grid_size = 32  
        self.goal_position = np.array([self.grid_size // 2, self.grid_size -1])  # Example goal position
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: A numpy array containing the current observation.
                Format: [x, y, vx, vy, wx, wy] where:
                - (x, y) is the current position
                - (vx, vy) is the current velocity
                - (wx, wy) is the current wind vector
        
        Returns:
            action: An integer in [0, 8] representing the action to take:
                - 0: Move North
                - 1: Move Northeast
                - 2: Move East
                - 3: Move Southeast
                - 4: Move South
                - 5: Move Southwest
                - 6: Move West
                - 7: Move Northwest
                - 8: Stay in place
        """
        # Extract position and wind from observation
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        current_pos = np.array([x, y])
        wind_vec = np.array([wx, wy])
        wind_norm = np.linalg.norm(wind_vec)
        if wind_norm > 1e-8:
            wind_dir = wind_vec / wind_norm
        else:
            wind_dir = np.array([0.0, 0.0])

        # You must provide the goal position to the agent, e.g. via self.goal_position
        # For demonstration, let's assume self.goal_position exists and is set externally
        direction_to_goal = self.goal_position - current_pos
        direction_to_goal_norm = np.linalg.norm(direction_to_goal)
        if direction_to_goal_norm > 1e-8:
            direction_to_goal = direction_to_goal / direction_to_goal_norm
        else:
            direction_to_goal = np.array([0.0, 0.0])

        # Define possible directions and action map
        directions = np.array([
            [0, 1], [1, 1], [1, 0], [1, -1],
            [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 0]
        ])
        action_map = {
            (0,1): 0, (1,1): 1, (1,0): 2, (1,-1): 3,
            (0,-1): 4, (-1,-1): 5, (-1,0): 6, (-1,1): 7, (0,0): 8
        }

        # Compute scores for each direction
        scores = []
        for d in directions:
            d_norm = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-8 else d
            similarity = np.dot(d_norm, direction_to_goal)
            efficiency = self.calculate_sailing_efficiency(d_norm, wind_dir)
            score = 0.7 * similarity + 0.3 * efficiency
            scores.append(score)

        best_idx = np.argmax(scores)
        best_direction = tuple(directions[best_idx])
        action = action_map[best_direction]
        return action
    
    def reset(self) -> None:
        """Reset the agent's internal state between episodes."""
        # Nothing to reset for this simple agent
        pass
        
    def seed(self, seed: int = None) -> None:
        """Set the random seed for reproducibility."""
        self.np_random = np.random.default_rng(seed)
        
    def save(self, path: str) -> None:
        """
        Save the agent's learned parameters to a file.
        
        Args:
            path: Path to save the agent's state
        """
        # No parameters to save for this simple agent
        pass
        
    def load(self, path: str) -> None:
        """
        Load the agent's learned parameters from a file.
        
        Args:
            path: Path to load the agent's state from
        """
        # No parameters to load for this simple agent
        pass 
    
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
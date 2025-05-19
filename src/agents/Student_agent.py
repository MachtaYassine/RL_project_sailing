"""
Naive Agent for the Sailing Challenge

This file provides a simple agent that always goes north.
Students can use it as a reference for how to implement a sailing agent.
"""

import numpy as np # type: ignore
from agents.base_agent import BaseAgent

class StudentAgent(BaseAgent):
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
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.
        
        """
        return 
    
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
from agents.base_agent import BaseAgent
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple

class StudentAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        print("Loading model...")
        # self.load('student_agent_bc.pth')

    def _build_model(self):
        class DualHeadNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Shared encoder for observation
                self.obs_encoder = torch.nn.Sequential(
                    torch.nn.Linear(2054, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                )
                # Predict wind params from obs only
                self.wind_head = torch.nn.Linear(64, 13)
                # Action head takes obs encoding + wind params + step
                self.action_head = torch.nn.Sequential(
                    torch.nn.Linear(64 + 13 + 1, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 9)
                )

            def forward(self, obs, wind_params=None, step=None):
                x = self.obs_encoder(obs)
                pred_wind = self.wind_head(x)
                # For action: use predicted wind if not given
                if wind_params is None:
                    wind_params = pred_wind
                if step is None:
                    step = torch.zeros((obs.shape[0], 1), device=obs.device) if obs.ndim == 2 else torch.tensor([0.0], device=obs.device)
                # Concatenate for action head
                if obs.ndim == 1:
                    action_in = torch.cat([x, wind_params, step], dim=0)
                else:
                    action_in = torch.cat([x, wind_params, step], dim=1)
                action_logits = self.action_head(action_in)
                return action_logits, pred_wind

        return DualHeadNet()

    def act(self, observation: np.ndarray, wind_params: np.ndarray = None, step: int = 0) -> int:
        """
        Select an action based on the current observation, wind parameters, and step.
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
            if wind_params is not None:
                wind_tensor = torch.tensor(wind_params, dtype=torch.float32).to(self.device)
            else:
                wind_tensor = None
            step_tensor = torch.tensor([step], dtype=torch.float32).to(self.device)
            action_logits, _ = self.model(obs_tensor, wind_tensor, step_tensor)
            action = torch.argmax(action_logits).item()
        return action

    def predict_wind_params(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict wind parameters from the current observation.
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
            _, wind_params = self.model(obs_tensor)
            return wind_params.cpu().numpy()

    def save(self, path: str):
        """
        Save the model weights to the given path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load the model weights from the given path.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {path}")

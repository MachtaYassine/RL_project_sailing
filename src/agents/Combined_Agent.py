import torch
import torch.nn as nn
import numpy as np
from agents.base_agent import BaseAgent

class DQNPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)


class ConvPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CombinedStudentAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        
        self.load('/scratch/ymachta/comb_student_agent_bc.pth')  # Load pre-trained weights
        self.step = 0

    def _build_model(self):
        class CombinedDualHeadNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.dqn = DQNPredictor()
                self.cnn = ConvPredictor()
                self.attention = nn.Parameter(torch.tensor(0.5))
                self.fc = nn.Sequential(
                    nn.Linear(128 + 256 + 13 + 1, 128),  # +13 wind params, +1 step
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 9)
                )
                self.wind_head = nn.Linear(128 + 256, 13)

            def forward(self, obs, wind_params, step):
                # obs: (batch, 2054), wind_params: (batch, 13), step: (batch, 1)
                dqn_input = obs[:,:6]  
                cnn_input = obs[:, 6:].reshape(-1, 2, 32, 32)  # Next 2048 features as 2x32x32 grid
                dqn_out = self.dqn(dqn_input)
                cnn_out = self.cnn(cnn_input)
                alpha = torch.sigmoid(self.attention)
                combined = torch.cat([alpha * dqn_out, (1 - alpha) * cnn_out], dim=1)
                pred_wind = self.wind_head(combined)
                action_in = torch.cat([combined, wind_params, step], dim=1)
                action_logits = self.fc(action_in)
                return action_logits, pred_wind

        return CombinedDualHeadNet()

    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation only (inference mode).
        Uses predicted wind parameters and the agent's internal step.
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Predict wind params from obs
            # Prepare CNN and DQN inputs
            dqn_input = obs_tensor[:, :6]
            cnn_input = obs_tensor[:, 6:].reshape(-1, 2, 32, 32)
            dqn_out = self.model.dqn(dqn_input)
            cnn_out = self.model.cnn(cnn_input)
            alpha = torch.sigmoid(self.model.attention)
            combined = torch.cat([alpha * dqn_out, (1 - alpha) * cnn_out], dim=1)
            pred_wind = self.model.wind_head(combined)
            step_tensor = torch.tensor([[self.step]], dtype=torch.float32).to(self.device)
            # Action head uses predicted wind and current step
            action_in = torch.cat([combined, pred_wind, step_tensor], dim=1)
            action_logits = self.model.fc(action_in)
            action = torch.argmax(action_logits, dim=1).item()
        self.step += 1
        return action

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {path}")
        
    def reset(self):
        """
        Reset the agent's state.
        """
        self.step=0
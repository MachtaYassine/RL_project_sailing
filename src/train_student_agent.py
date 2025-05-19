import numpy as np
import torch
import torch.optim as optim
import os
import sys
import random
sys.path.append("..")
sys.path.append(".")

from agents.Student_agent import StudentAgent
from agents.Deterministic_agent2 import DeterministicAgent2
from env_sailing import SailingEnv
from initial_windfields import INITIAL_WINDFIELDS
from tqdm import tqdm

# --- Hyperparameters ---
NUM_EPISODES_PER_WINDFIELD = 100
MAX_STEPS_PER_EPISODE = 200
BATCH_SIZE = 512
EPOCHS = 1000
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "student_agent_bc.pth"
EXPERT_DATA_PATH = "expert_data.npz"
CHECK_EXPERT_SUCCESS_STRIDE = 50  # Print stats every X episodes

def validate_student(student, windfield_names, max_steps=200):
    student.model.eval()
    results = []
    for windfield_name in windfield_names:
        windfield = INITIAL_WINDFIELDS[windfield_name]
        env = SailingEnv(
            wind_init_params=windfield['wind_init_params'],
            wind_evol_params=windfield['wind_evol_params']
        )
        seed = random.randint(0, 100000)
        obs, _ = env.reset(seed=seed)
        student.reset()
        steps = 0
        success = False
        for _ in range(max_steps):
            action = student.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            if done:
                success = True
                break
            if truncated:
                break
        results.append((windfield_name, success, steps))
    print("Validation results:")
    for windfield_name, success, steps in results:
        print(f"  {windfield_name}: {'Success' if success else 'Fail'} in {steps} steps")
    student.model.train()

# --- Data Collection ---
def collect_expert_data(agent, windfield, num_episodes, max_steps, check_expert_success_stride=CHECK_EXPERT_SUCCESS_STRIDE):
    X, y = [], []
    episode_success = []
    episode_steps = []
    for ep in tqdm(range(num_episodes), desc="Collecting expert data"):
        env = SailingEnv(
            wind_init_params=windfield['wind_init_params'],
            wind_evol_params=windfield['wind_evol_params']
        )
        obs, _ = env.reset(seed=ep)
        agent.reset()
        steps = 0
        success = False
        for _ in range(max_steps):
            action = agent.act(obs)
            X.append(obs.copy())  # Use the full observation!
            y.append(action)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            if done:
                success = True
                break
            if truncated:
                break
        episode_success.append(success)
        episode_steps.append(steps)
        # Print stats every check_expert_success_stride episodes
        if (ep + 1) % check_expert_success_stride == 0:
            recent_success = episode_success[-check_expert_success_stride:]
            recent_steps = episode_steps[-check_expert_success_stride:]
            success_rate = 100.0 * np.mean(recent_success)
            avg_steps = np.mean(recent_steps)
            print(f"[Ep {ep+1}] Success rate: {success_rate:.1f}% | Avg steps: {avg_steps:.1f}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# --- Main Training Script ---
def main():
    # 1. Load or collect expert data
    if os.path.exists(EXPERT_DATA_PATH):
        print(f"Loading expert data from {EXPERT_DATA_PATH}")
        data = np.load(EXPERT_DATA_PATH)
        X = data["X"]
        y = data["y"]
    else:
        expert = DeterministicAgent2()
        all_X, all_y = [], []
        for windfield_name in ["training_1", "training_2", "training_3", "training_4", "training_5", "training_6", "training_7", "training_8"]:
            print(f"Collecting data for windfield: {windfield_name}")
            windfield = INITIAL_WINDFIELDS[windfield_name]
            X_part, y_part = collect_expert_data(
                expert, windfield, NUM_EPISODES_PER_WINDFIELD, MAX_STEPS_PER_EPISODE,
                check_expert_success_stride=CHECK_EXPERT_SUCCESS_STRIDE
            )
            all_X.append(X_part)
            all_y.append(y_part)
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        print(f"Total samples collected: {X.shape[0]}")
        np.savez(EXPERT_DATA_PATH, X=X, y=y)
        print(f"Expert data saved to {EXPERT_DATA_PATH}")

    # 2. Prepare PyTorch dataset and dataloader
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize StudentAgent and optimizer
    student = StudentAgent()
    student.model.train()
    optimizer = optim.Adam(student.model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Training loop
    windfield_names = ["training_1", "training_2", "training_3", "training_4", "training_5", "training_6", "training_7", "training_8"]
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in tqdm(dataloader):
            batch_X = batch_X.to(student.device)
            batch_y = batch_y.to(student.device)
            optimizer.zero_grad()
            logits = student.model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Validation after epoch {epoch+1}:")
            validate_student(student, windfield_names, max_steps=MAX_STEPS_PER_EPISODE)

    # 5. Save the trained model
    student.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
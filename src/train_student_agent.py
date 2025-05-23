import numpy as np
import torch
import torch.optim as optim
import os
import sys
import random
import matplotlib.pyplot as plt
sys.path.append("..")
sys.path.append(".")

from src.agents.Student_agent import StudentAgent
from agents.Deterministic_agent2 import DeterministicAgent2
from env_sailing import SailingEnv
from initial_windfields import INITIAL_WINDFIELDS
from tqdm import tqdm

# Add import for windfield sampling
from src.initial_windfields.sample_windfields import sample_windfield


# --- Hyperparameters ---
NUM_EPISODES_PER_WINDFIELD = 100
MAX_STEPS_PER_EPISODE = 200
BATCH_SIZE = 512
EPOCHS = 1000
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "student_agent_bc.pth"
EXPERT_DATA_PATH = "expert_data.npz"
CHECK_EXPERT_SUCCESS_STRIDE = 50  # Print stats every X epochs

TRAIN_WF = ["training_1", "training_2", "training_3", "training_4", "training_5"]
VAL_WF = ["training_6", "training_7", "training_8"]

def validate_student(student, windfield_names, max_steps=200, episodes_per_wf=10):
    """
    Evaluate the student agent on a set of windfields.

    Returns:
        success_rates: List of success rates per windfield (float in [0,1])
        avg_steps: List of average steps per windfield (float)
    """
    student.model.eval()
    success_rates = []
    avg_steps = []
    avg_reward = []
    for windfield_name in windfield_names:
        windfield = INITIAL_WINDFIELDS[windfield_name]
        successes = []
        steps_list = []
        rewards = []
        for ep in range(episodes_per_wf):
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
            successes.append(success)
            steps_list.append(steps)
            rewards.append(reward* 0.99 ** steps)
        avg_reward.append(np.mean(rewards))
        success_rate = np.mean(successes)
        avg_step = np.mean(steps_list)
        success_rates.append(success_rate)
        avg_steps.append(avg_step)
    student.model.train()
    return success_rates, avg_steps , avg_reward

def collect_expert_data_sampled(agent, num_windfields, seeds_per_wf, max_steps, check_expert_success_stride=CHECK_EXPERT_SUCCESS_STRIDE):
    """
    Collect expert data using randomly sampled windfields.
    For each windfield, run seeds_per_wf episodes with different random seeds.
    Prints stats for each sampled windfield (across the number of seeds).
    """
    X, y = [], []
    for wf_idx in range(num_windfields):
        windfield = sample_windfield()
        wf_success = []
        wf_steps = []
        wf_rewards = []
        for seed in range(seeds_per_wf):
            env = SailingEnv(
                wind_init_params=windfield['wind_init_params'],
                wind_evol_params=windfield['wind_evol_params']
            )
            obs, _ = env.reset(seed=seed)
            agent.reset()
            steps = 0
            success = False
            for _ in range(max_steps):
                action = agent.act(obs)
                X.append(obs.copy())
                y.append(action)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                if done:
                    success = True
                    break
                if truncated:
                    break
            wf_success.append(success)
            wf_steps.append(steps)
            wf_rewards.append(reward * 0.99 ** steps)
        # Print stats for this windfield
        success_rate = 100.0 * np.mean(wf_success)
        avg_steps = np.mean(wf_steps)
        avg_reward = np.mean(wf_rewards)
        print(f"[Windfield {wf_idx+1}/{num_windfields}] Success rate: {success_rate:.1f}% | Avg steps: {avg_steps:.1f} | Avg reward: {avg_reward:.2f}")
        # if avg_reward < 55:
            # print(f" windfield init parameters: {windfield['wind_init_params']} \n windfield evol params: {windfield['wind_evol_params']}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)



def main():
    # 1. Load or collect expert data (only for training windfields)
    # 1. Load or collect expert data (using sampled windfields)
    if os.path.exists(EXPERT_DATA_PATH):
        print(f"Loading expert data from {EXPERT_DATA_PATH}")
        data = np.load(EXPERT_DATA_PATH)
        X = data["X"]
        y = data["y"]
    else:
        expert = DeterministicAgent2()
        # Use 500 windfields, 5 seeds each, 100 steps max (as per your requirements)
        X, y = collect_expert_data_sampled(
            expert, num_windfields=500, seeds_per_wf=5, max_steps=100,
            check_expert_success_stride=CHECK_EXPERT_SUCCESS_STRIDE
        )
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

    # 4. Training loop with validation
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    train_losses = []
    val_success_rates = []
    val_avg_steps = []
    val_rewards = []  # <-- initialize as list!
    val_x = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_X = batch_X.to(student.device)
            batch_y = batch_y.to(student.device)
            optimizer.zero_grad()
            logits = student.model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        avg_loss = total_loss / len(dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

        # Validation and live plotting every CHECK_EXPERT_SUCCESS_STRIDE epochs
        if (epoch + 1) % CHECK_EXPERT_SUCCESS_STRIDE == 0:
            print(f"Validation after epoch {epoch+1}:")
            val_rates, val_steps, val_rewards_epoch = validate_student(
                student, VAL_WF, max_steps=MAX_STEPS_PER_EPISODE, episodes_per_wf=10
            )
            mean_val_rate = np.mean(val_rates)
            mean_val_steps = np.mean(val_steps)
            mean_val_reward = np.mean(val_rewards_epoch)
            val_success_rates.append(mean_val_rate)
            val_avg_steps.append(mean_val_steps)
            val_rewards.append(mean_val_reward)
            val_x.append(epoch + 1)
            print(f"Validation success rates (windfields 6-8): {val_rates}, mean: {mean_val_rate:.2f}")
            print(f"Validation avg steps (windfields 6-8): {val_steps}, mean: {mean_val_steps:.2f}")

            # --- Live Plotting ---
            axs[0].cla()
            axs[0].plot(train_losses, label="Train Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("Training Loss")
            axs[0].legend()

            axs[1].cla()
            axs[1].plot(val_x, val_success_rates, marker='o', label="Validation Success Rate")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Success Rate")
            axs[1].set_title("Validation Success Rate (Windfields 6-8)")
            axs[1].set_ylim(0, 1.05)
            axs[1].legend()

            axs[2].cla()
            axs[2].plot(val_x, val_avg_steps, marker='o', label="Validation Avg Steps")
            axs[2].set_xlabel("Epoch")
            axs[2].set_ylabel("Avg Steps")
            axs[2].set_title("Validation Avg Steps (Windfields 6-8)")
            axs[2].legend()

            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

    # 5. Save the trained model
    student.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
    print('\a')
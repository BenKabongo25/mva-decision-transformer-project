# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import enum
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load
from datasets import StateActionReturnToGoTimeDataset, StateActionRewardTimeDataset
from models.rnn_model import RNN4RL
from models.transformer_models import DecisionTransformer, BehaviorCloning


plt.style.use("bmh")
warnings.filterwarnings(action="ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelName(enum.Enum):
    DECISION_TRANSFORMER = 0
    BEHAVIOR_CLOSING = 1
    RNN4RL = 2


class Config:

    def __init__(self, data_dir, model_name=ModelName.DECISION_TRANSFORMER):
        self.data_dir=data_dir          
        self.model_name=model_name      # RNN or Transformer model
        self.max_length=12
        self.batch_size=128
        self.n_heads=8                  # Transformers models
        self.n_blocks=6                 # Transformers models
        self.embedding_dim=128
        self.hidden_size=64             # RNN model
        self.project_size=32            # RNN model
        self.embedding_pdropout=0.0
        self.attention_pdropout=0.0     # Transformers models
        self.output_pdropout=0.0        # Transformers models
        self.lr=6e-4
        self.weight_decay=0
        self.betas=(0.9, 0.95)
        self.num_epochs=10
        self.seed=42
        self.max_step=1_000             # eval
        self.plot_dir="."               # 
        self.experiment_name="run"      #
        self.verbose=False
        self.rtg=True                   # rtg or reward dataset


def train(model, dataloader, optimizer, config):
    model.to(device)
    model.train()

    all_loss = []
    loss_fn = nn.CrossEntropyLoss()

    progress_bar = range(config.num_epochs)
    if config.verbose:
        progress_bar = tqdm(progress_bar, "Training")

    for epoch in progress_bar:
        total_loss = 0.0

        for states, actions, rewards, timesteps, targets in dataloader:
            states, actions, rewards, timesteps, targets = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
                timesteps.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            predictions = model(states, actions, rewards, timesteps)
            loss = loss_fn(predictions.view(-1, model.n_actions), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        if config.verbose:
            progress_bar.set_description(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {average_loss:.4f}")
        all_loss.append(average_loss)

    return all_loss


def evaluate(factory, model, metadata, config):
    target_return = metadata["target_return"]
    state, info = factory.model.reset(seed=config.seed)
    state = factory.model._current_state
    action = factory.get_policy()[state]
    next_state, reward, terminated, truncated, info = factory.model.step(action)

    states, actions, rtgs, rewards, times, all_rewards = [], [], [], [], [], []
    if config.verbose:
        print("Action", "=>", "State", "Reward", "Done", "Trunc.", "Info", "Score", sep="\t") 
        print(".\t =>", state, None, False, False, info, sep="\t")
        print(action, "=>", next_state, reward, terminated, truncated, info, sep="\t")
    state = next_state

    for t in range(config.max_step):
        states.append(int(state))
        actions.append(int(action))
        rewards.append(float(reward))
        all_rewards.append(float(reward))
        times.append(int(t))

        if t == 0:
            rtgs.append(target_return - reward)
        else:
            rtgs.append(rtgs[-1] - reward)

        if len(states) >= config.max_length // 3:
            states = states[-config.max_length // 3:]
            actions = actions[-config.max_length // 3:]
            rtgs = rtgs[-config.max_length // 3:]
            rewards = rewards[-config.max_length // 3:]
            times = times[-config.max_length // 3:]

        rewards_ = rtgs if config.rtg else rewards

        states_, actions_, rewards_, times_ = (
            torch.LongTensor(states).reshape(1, -1).to(device), 
            torch.LongTensor(actions).reshape(1, -1).to(device),
            torch.FloatTensor(rewards_).reshape(1, -1).to(device),
            torch.FloatTensor(times).reshape(1, -1).to(device)
        )

        action =  model(states_, actions_, rewards_, times_).argmax(dim=-1)[0, -1].item()
        next_state, reward, terminated, truncated, info = factory.model.step(action)
        state = next_state
        if config.verbose:
            print(action, "=>", next_state, reward, terminated, truncated, info, sum(rewards), sep="\t")
        if terminated or truncated:
            break

    score = sum(all_rewards)
    return score


def normalize_score(score, min_score, max_score):
    return ((score - min_score) / (max_score - min_score)) * 100


def run(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    factory, metadata, data = load(config.data_dir)

    n_states = metadata["n_states"]
    n_actions = metadata["n_actions"]
    target_return = metadata["target_return"]
    max_t = metadata["data_max_step"]

    if config.rtg:
        dataset = StateActionReturnToGoTimeDataset(data, max_length=config.max_length, n_states=n_states, n_actions=n_actions)
    else:
        dataset = StateActionRewardTimeDataset(data, max_length=config.max_length, n_states=n_states, n_actions=n_actions)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if config.model_name is ModelName.RNN4RL:
        model = RNN4RL(n_states, n_actions, embedding_dim=config.embedding_dim, hidden_size=config.hidden_size, 
                       project_size=config.project_size, embedding_pdropout=config.embedding_pdropout)
    model = DecisionTransformer(n_states, n_actions, max_length=config.max_length, max_t=max_t, 
                            n_heads=config.n_heads, n_blocks=config.n_blocks, embedding_dim=config.embedding_dim, device=device)
    optimizer = model.configure_optimizers(lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)

    all_loss = train(model, dataloader, optimizer, num_epochs=config.num_epochs)
    os.makedirs(config.plot_dir, exist_ok=True)
    plt.plot(all_loss)
    plt.savefig(config.plot_dir + f"/{config.experiment_name}_loss.png")

    factory.train_policy()
    score = evaluate(factory, model, metadata, config)
    min_score = config.max_step * min(factory.model.all_rewards)
    max_score = target_return
    normalized_score = normalize_score(score, min_score, max_score)
    return normalized_score


def main():
    base_dir = "../data/mdp"
    res_dir = "../results"
    os.makedirs(res_dir)

    results = {}

    for model_name in tqdm(( ModelName.DECISION_TRANSFORMER,
                        ModelName.BEHAVIOR_CLOSING,
                        ModelName.RNN4RL ), "Model"):

        model_name = model_name.name
        results[model_name] = {}

        if model_name is ModelName.BEHAVIOR_CLOSING: # rtg/reward indifferent
            rtgs = (True,)
        else:
            rtgs = (True, False)

        for rtg in tqdm(rtgs, f"{model_name} rtg"):
            rtg_name = "rtg" if rtg else "reward"
            results[model_name][rtg_name] = {}

            for n_states in tqdm([10, 100, 1_000, 10_000, 100_000, 1_000_000], f"{model_name} {rtg_name} States"):
                s_name = f"S{n_states}"
                results[model_name][rtg_name][s_name] = {}

                for n_actions in tqdm([2, 3, 4, 5, 10, 20, 50, 100], f"{model_name} {rtg_name} {s_name} Actions"):
                    a_name = f"A{n_actions}"
                    results[model_name][rtg_name][s_name][a_name] = {}

                    for n_rewards in tqdm([2, 3, 4, 5, 10], f"{model_name} {rtg_name} {s_name} {a_name} Rewards"):
                        r_name = f"R{n_rewards}"

                        experiment_name = f"{model_name}_{rtg_name}_{s_name}_{a_name}_{r_name}"
                        data_dir_name = f"S{n_states}_A{n_actions}_R{n_rewards}_T2_R1"
                        if not os.path.exists(data_dir_name):
                            print(f"Not found : {data_dir_name}")
                            continue

                        data_dir = f"{base_dir}/{data_dir_name}/"
                        config = Config(data_dir, model_name)
                        config.rtg = rtg
                        config.plot_dir = f"{res_dir}/plots"
                        config.experiment_name = experiment_name
                        score = run(config)

                        results[model_name][rtg_name][s_name][a_name][r_name] = score
                        with open(res_dir + "/results.json", "w") as file:
                            json.dump(results, file)


if __name__ == "__main__":
    main()

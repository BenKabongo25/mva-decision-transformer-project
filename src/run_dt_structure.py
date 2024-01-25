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
from datasets import StateActionReturnToGoTimeDataset
from models.transformer_models import DecisionTransformer


plt.style.use("bmh")
warnings.filterwarnings(action="ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]:", device)



class Config:

    def __init__(self):        
        self.max_length=12
        self.batch_size=128
        self.n_heads=8
        self.n_blocks=6
        self.embedding_dim=128
        self.embedding_pdropout=0.0
        self.attention_pdropout=0.0
        self.output_pdropout=0.0
        self.lr=6e-4
        self.weight_decay=0
        self.betas=(0.9, 0.95)
        self.num_epochs=5
        self.seed=42
        self.max_step=1_000
        self.plot_dir="."               
        self.experiment_name="run" 
        self.verbose=False
        self.factory=None
        self.metadata=None
        self.data=None
        self.n_actions=0
        self.n_states=0
        self.n_rewards=0
        self.max_t=0


def train(model, dataloader, optimizer, config):
    model.to(device)
    model.train()

    all_loss = []
    loss_fn = nn.CrossEntropyLoss()

    progress_bar = range(config.num_epochs)
    if config.verbose:
        progress_bar = tqdm(progress_bar, config.experiment_name)

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
            progress_bar.set_description(f"[{config.experiment_name}] Epoch {epoch + 1}/{config.num_epochs}, Loss: {average_loss:.4f}")
        all_loss.append(average_loss)

    return all_loss


def evaluate(model, config):
    target_return = config.metadata["target_return"]
    state, info = config.factory.model.reset(seed=config.seed)
    state = config.factory.model._current_state
    action = config.factory.get_policy()[state]
    next_state, reward, terminated, truncated, info = config.factory.model.step(action)

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

        rewards_ = rtgs
        states_, actions_, rewards_, times_ = (
            torch.LongTensor(states).reshape(1, -1).to(device), 
            torch.LongTensor(actions).reshape(1, -1).to(device),
            torch.FloatTensor(rewards_).reshape(1, -1).to(device),
            torch.FloatTensor(times).reshape(1, -1).to(device)
        )

        action =  model(states_, actions_, rewards_, times_).argmax(dim=-1)[0, -1].item()
        next_state, reward, terminated, truncated, info = config.factory.model.step(action)
        state = next_state
        if config.verbose:
            print(action, "=>", next_state, reward, terminated, truncated, info, sum(all_rewards), sep="\t")
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

    dataset = StateActionReturnToGoTimeDataset(
        config.data, 
        max_length=config.max_length, 
        n_states=config.n_states, 
        n_actions=config.n_actions
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = DecisionTransformer(
        config.n_states, 
        config.n_actions, 
        max_length=config.max_length, 
        max_t=config.max_t, 
        n_heads=config.n_heads, 
        n_blocks=config.n_blocks, 
        embedding_dim=config.embedding_dim, 
        device=device
    )
    optimizer = model.configure_optimizers(lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)

    all_loss = train(model, dataloader, optimizer, config)
    os.makedirs(config.plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(all_loss)
    plt.savefig(config.plot_dir + f"/{config.experiment_name}_loss.png")

    #factory.train_policy()
    score = evaluate(model, config)
    min_score = config.max_step * min(config.factory.model.all_rewards)
    max_score = config.metadata["target_return"]
    normalized_score = normalize_score(score, min_score, max_score)
    return normalized_score


def main():
    data_dir = "../data/mdp/S1000_A10_R5_T2_R1/" # 1000 S, 10 A, 5 R

    factory, metadata, data = load(data_dir)
    n_states = metadata["n_states"]
    n_actions = metadata["n_actions"]
    n_rewards = metadata["n_rewards"]
    max_t = metadata["data_max_step"]

    res_dir = "../results_dt_structure"
    os.makedirs(res_dir, exist_ok=True)

    results = {}
    if os.path.exists(res_dir + "/results.json"):
        with open(res_dir + "/results.json", "r") as file:
            results = json.load(file)
            print(results)

    for n_blocks in tqdm([1, 2, 4, 6, 8], "Main"):

        n_blocks_name = f"B{n_blocks}"
        if n_blocks_name not in results:
            results[n_blocks_name] = {}

        for n_heads in tqdm([1, 2, 4, 8], n_blocks_name):

            n_heads_name = f"H{n_heads}"
            if n_heads_name not in results[n_blocks_name]:
                results[n_blocks_name][n_heads_name] = {}

            for embedding_dim in tqdm([16, 32, 64, 128, 256], n_heads_name):

                dim_name = f"D{embedding_dim}"
                if dim_name not in results[n_blocks_name][n_heads_name]:
                    
                    experiment_name = f"{n_blocks_name}_{n_heads_name}_{dim_name}"
                    config = Config()
                    config.factory = factory
                    config.metadata = metadata
                    config.data = data
                    config.n_states = n_states
                    config.n_actions = n_actions
                    config.n_rewards = n_rewards
                    config.max_t = max_t
                    config.n_blocks = n_blocks
                    config.n_heads = n_heads
                    config.embedding_dim = embedding_dim
                    config.embedding_pdropout = .1
                    config.attention_pdropout = .1
                    config.plot_dir = f"{res_dir}/plots"
                    config.experiment_name = experiment_name
                    score = run(config)

                    results[n_blocks_name][n_heads_name][dim_name] = score
                    with open(res_dir + "/results.json", "w") as file:
                        json.dump(results, file)


if __name__ == "__main__":
    main()

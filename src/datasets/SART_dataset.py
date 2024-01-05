# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np
import torch


class SARTDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_length, n_states, n_actions):        
        self.max_length = max_length
        self.n_states = n_states
        self.n_actions = n_actions
        self.data = []
        self.done_idxs = []
        for (states, actions, rewards, times) in data:
            if len(states) > max_length:
                rtgs = (np.cumsum(rewards[::-1])[::-1]).tolist()
                item = [states, actions, rtgs, times]
                self.data.append(item)
                self.done_idxs.append(len(states) - max_length)
        self.done_idxs = np.cumsum(self.done_idxs)
    

    def __len__(self):
        return self.done_idxs[-1]
        

    def __getitem__(self, idx):
        item_idx = 0
        for i, done_idx in enumerate(self.done_idxs):
            if done_idx > idx:
                item_idx = i
                break
        idx = self.done_idxs[item_idx] - idx
        states  = self.data[item_idx][0][idx: idx + self.max_length]
        actions = self.data[item_idx][1][idx: idx + self.max_length]
        targets = self.data[item_idx][1][idx + self.max_length]
        rtgs    = self.data[item_idx][2][idx: idx + self.max_length]
        times   = self.data[item_idx][3][idx: idx + self.max_length]
        return (states, actions, rtgs, times), targets
        
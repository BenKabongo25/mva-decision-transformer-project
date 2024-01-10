# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np
import torch


class StateActionReturnToGoTimeDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_length, n_states, n_actions):        
        self.max_length = max_length // 3
        self.n_states = n_states
        self.n_actions = n_actions
        self.data = []
        self.done_idxs = []
        for (states, actions, rewards, times) in data:
            if len(states) >= max_length + 1:
                rtgs = (np.cumsum(rewards[::-1])[::-1]).tolist()
                assert len(rtgs) == len(states)
                item = [states, actions, rtgs, times]
                self.data.append(item)
                self.done_idxs.append(len(states) - self.max_length - 1)
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
        states  = torch.LongTensor(self.data[item_idx][0][idx: idx + self.max_length])
        actions = torch.LongTensor(self.data[item_idx][1][idx: idx + self.max_length])
        targets = torch.LongTensor(self.data[item_idx][1][1 + idx: 1 + idx + self.max_length])
        rtgs    = torch.FloatTensor(self.data[item_idx][2][idx: idx + self.max_length])
        times   = torch.FloatTensor(self.data[item_idx][3][idx: idx + self.max_length])
        return states, actions, rtgs, times, targets
        

class StateActionRewardTimeDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_length, n_states, n_actions):        
        self.max_length = max_length // 3
        self.n_states = n_states
        self.n_actions = n_actions
        self.data = []
        self.done_idxs = []
        for (states, actions, rewards, times) in data:
            if len(states) >= max_length + 1:
                item = [states, actions, rewards, times]
                self.data.append(item)
                self.done_idxs.append(len(states) - self.max_length - 1)
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
        states  = torch.LongTensor(self.data[item_idx][0][idx: idx + self.max_length])
        actions = torch.LongTensor(self.data[item_idx][1][idx: idx + self.max_length])
        targets = torch.LongTensor(self.data[item_idx][1][1 + idx: 1 + idx + self.max_length])
        rewards = torch.FloatTensor(self.data[item_idx][2][idx: idx + self.max_length])
        times   = torch.FloatTensor(self.data[item_idx][3][idx: idx + self.max_length])
        return states, actions, rewards, times, targets
        

class StateActionRewardStateDataset(torch.utils.data.Dataset):

    def __init__(self, data, n_states, n_actions):    
        self.n_states = n_states
        self.n_actions = n_actions
        self.data = []
        self.done_idxs = []
        for (states, actions, rewards, _) in data:
            if len(states) >= 2:
                item = [states, actions, rewards]
                self.data.append(item)
                self.done_idxs.append(len(states) - 2 - 1)
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
        states   = torch.LongTensor(self.data[item_idx][0][idx])
        n_states = torch.LongTensor(self.data[item_idx][0][idx + 1])
        actions  = torch.LongTensor(self.data[item_idx][1][idx])
        rewards  = torch.FloatTensor(self.data[item_idx][2][idx])
        return states, actions, rewards, n_states
        
# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np

import mdp
from policies import MDPPolicy


class TDPolicy(MDPPolicy):

    def __init__(self, model: mdp.MDP, alpha: float, gamma: float=0.99, eps: float=1e-3):
        super().__init__(model)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = np.zeros((self.model.config.n_states, self.model.config.n_actions))

    
    def fit(self, n_episodes: int=100):
        raise NotImplementedError


    def choose_action(self, s: int=None) -> int:
        if np.random.random() < self.eps:
            a = np.random.choice(np.arange(self.model.config.n_actions))
        else:
            a = np.argmax(self.Q[s, :])
        return a


    def get_policy(self):
        return np.argmax(self.Q, axis=1)


class SARSA(TDPolicy):

    def fit(self, n_episodes=100):
        for _ in range(n_episodes):
            s, _ = self.model.reset()
            a = self.choose_action(s)
            done = False
            while not done:
                next_s, r, terminated, truncated, info = self.model.step(a)
                next_a = self.choose_action(next_s)
                self.Q[s, a] = self.Q[s, a] + self.alpha * (r + self.gamma * self.Q[next_s, next_a] - self.Q[s, a])
                s = next_s 
                a = next_a
                done = terminated or truncated


class QLearning(TDPolicy):

    def fit(self, n_episodes=100):
        for _ in range(n_episodes):
            s, _ = self.model.reset()
            done = False
            while not done:
                a = self.choose_action(s)
                next_s, r, terminated, truncated, info = self.model.step(a)
                self.Q[s, a] = self.Q[s, a] + self.alpha * (r + self.gamma * np.max(self.Q[next_s, :]) - self.Q[s, a])
                s = next_s
                done = terminated or truncated


class DoubleQLearning(TDPolicy):

    def __init__(self, model: mdp.MDP, alpha: float, gamma: float=0.99, eps: float=1e-3):
        super().__init__(model, alpha, gamma, eps)
        self.Q_ = np.zeros((self.model.config.n_states, self.model.config.n_actions))


    def fit(self, n_episodes=100):
        for _ in range(n_episodes):
            s, _ = self.model.reset()
            done = False
            while not done:
                a = self.choose_action(s)
                next_s, r, terminated, truncated, info = self.model.step(a)
                if np.random.random() < .5:
                    self.Q[s, a] = self.Q[s, a] + self.alpha * (
                        r + self.gamma * self.Q_[s, np.argmax(self.Q[next_s, :])] - self.Q[s, a])
                else:
                    self.Q_[s, a] = self.Q_[s, a] + self.alpha * (
                        r + self.gamma * self.Q[s, np.argmax(self.Q_[next_s, :])] - self.Q_[s, a])
                s = next_s
                done = terminated or truncated


    def choose_action(self, s: int=None) -> int:
        if np.random.random() < self.eps:
            a = np.random.choice(np.arange(self.model.config.n_actions))
        else:
            a = np.argmax(self.Q[s, :] + self.Q_[s, :])
        return a

    
    def get_policy(self):
        return np.argmax(self.Q + self.Q_, axis=1)
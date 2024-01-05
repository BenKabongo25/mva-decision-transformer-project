# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import json
import numpy as np
from collections import namedtuple

from enums import MDPTransitionType, MDPRewardType, SpaceType, PolicyType
from mdp import MDP, MDPConfig
from policies import VI, PI
from td_policies import SARSA, QLearning, DoubleQLearning
from utils import terminate_s, transition, reward
from wrappers import DiscreteActionWrapper, DiscreteObservationWrapper, Range


class MDPFactory(object):

    Data = namedtuple(
        "Data", 
        ["transition_function_type", "reward_function_type",
        "n_states", "n_actions", "n_rewards",
        "gamma", "eps", "alpha", "policy_type",
        "terminate_s_flags", "transitions", "rewards","all_rewards"]
    )


    def __init__(self, 
            transition_function_type=None, reward_function_type=None, 
            n_states=3, n_actions=2, n_rewards=2, 
            config=None, terminate_s_flags=None, transitions=None, rewards=None, all_rewards=None,
            p=0.2, gamma=0.99, eps=1e-3, alpha=1e-3,
            policy_type=PolicyType.VI
        ):
        if config is None:
            assert transition_function_type is not None
            assert reward_function_type is not None
            config = MDPConfig(
                state_space_type=SpaceType.DISCRETE,
                action_space_type=SpaceType.DISCRETE,
                transition_function_type=transition_function_type,
                reward_function_type=reward_function_type,
                n_states=n_states,
                n_actions=n_actions
            )
        
        if terminate_s_flags is None:
            terminate_s_flags = terminate_s(n_states, p=p)
        if transitions is None:
            transitions = transition(config.transition_function_type, n_states, n_actions)
        if all_rewards is None:
            all_rewards = np.arange(-n_rewards + 2, 2, 1)
        if rewards is None:
            rewards = reward(config.reward_function_type, n_states, n_actions, 
                            terminate_s_flags, transitions, all_rewards)
        
        self.init(config, terminate_s_flags, transitions, rewards, all_rewards,
                gamma=gamma, eps=eps, alpha=alpha,
                policy_type=policy_type)     


    def init(self, config, terminate_s_flags, transitions, rewards, all_rewards,
            gamma=0.99, eps=1e-3, alpha=1e-3,
            policy_type=PolicyType.VI
        ):
        self.model = MDP(config)

        self.terminate_s_flags = terminate_s_flags
        self.transitions = transitions
        self.rewards = rewards
        self.all_rewards = all_rewards

        terminate_function = lambda s: self.terminate_s_flags[s]

        def transition_function(s, a, next_s):
            if config.transition_function_type is MDPTransitionType.S_DETERMINISTIC:
                return int(self.transitions[s])

            if config.transition_function_type is MDPTransitionType.S_PROBABILISTIC:
                probas = self.transitions[s]
                next_ss = np.arange(len(probas))
                return dict(zip(next_ss, probas))

            if config.transition_function_type is MDPTransitionType.SA_DETERMINISTIC:
                return int(self.transitions[s, a])

            if config.transition_function_type is MDPTransitionType.SA_PROBABILISTIC:
                probas = self.transitions[s, a]
                next_ss = np.arange(len(probas))
                return dict(zip(next_ss, probas))

            if config.transition_function_type is MDPTransitionType.SAS:
                return self.transitions[s, a, next_s]

            raise ValueError("Unknown transition type")

        def reward_function(s, a, next_s, r):
            if config.reward_function_type is MDPRewardType.S:
                return self.rewards[s]

            if config.reward_function_type is MDPRewardType.SA:
                return self.rewards[s, a]

            if config.reward_function_type is MDPRewardType.SAS:
                return self.rewards[s, a, s]

            if config.reward_function_type is MDPRewardType.SASR:
                i = list(self.all_rewards).index(r)
                return self.rewards[s, a, next_s, i]
            
            raise ValueError("Unknown reward type")
            
        observation_wrapper = DiscreteObservationWrapper(self.model, Range(config.n_states))
        action_wrapper = DiscreteActionWrapper(self.model, Range(config.n_actions))

        self.model.init(
            observation_wrapper, 
            action_wrapper, 
            transition_function, 
            reward_function, 
            terminate_function, 
            self.all_rewards
        )

        if policy_type is PolicyType.VI:
            self.policy = VI(self.model, gamma, eps)
        elif policy_type is PolicyType.PI:
            self.policy = PI(self.model, gamma, eps)
        elif policy_type is PolicyType.SARSA:
            self.policy = SARSA(self.model, alpha, gamma, eps)
        elif policy_type is PolicyType.QLEARNING:
            self.policy = QLearning(self.model, alpha, gamma, eps)
        elif policy_type is PolicyType.DQLEARNING:
            self.policy = DoubleQLearning(self.model, alpha, gamma, eps)
        else:
            raise ValueError("Unknown policy type")
        self.policy_type = policy_type


    @classmethod
    def new(cls, transition_function_type, reward_function_type, 
            n_states, n_actions, n_rewards, 
            p=0.2, gamma=0.99, eps=1e-3, alpha=1e-3,
            policy_type=PolicyType.VI
        ):
        config = MDPConfig(
            state_space_type=SpaceType.DISCRETE,
            action_space_type=SpaceType.DISCRETE,
            transition_function_type=transition_function_type,
            reward_function_type=reward_function_type,
            n_states=n_states,
            n_actions=n_actions
        )

        terminate_s_flags = terminate_s(n_states, p=p)
        transitions = transition(config.transition_function_type, n_states, n_actions)
        all_rewards = np.arange(-n_rewards + 2, 2, 1)
        rewards = reward(config.reward_function_type, n_states, n_actions, 
                              terminate_s_flags, transitions, all_rewards)

        return MDPFactory(
            config=config, terminate_s_flags=terminate_s_flags, 
            transitions=transitions, rewards=rewards, all_rewards=all_rewards,
            gamma=gamma, eps=eps, alpha=alpha, policy_type=policy_type
        )


    @classmethod
    def load(cls, filename):
        data = dict()
        with open(filename, "r") as file:
            data = json.load(file)
        data = MDPFactory.Data(**data)
        
        config = MDPConfig(
            state_space_type=SpaceType.DISCRETE,
            action_space_type=SpaceType.DISCRETE,
            transition_function_type=MDPTransitionType(data.transition_function_type),
            reward_function_type=MDPRewardType(data.reward_function_type),
            n_states=data.n_states,
            n_actions=data.n_actions
        )

        terminate_s_flags = np.array(data.terminate_s_flags, dtype=bool)
        transitions = np.array(data.transitions)
        rewards = np.array(data.rewards)
        all_rewards = np.array(data.all_rewards)

        return MDPFactory(
            config=config, terminate_s_flags=terminate_s_flags, 
            transitions=transitions, rewards=rewards, all_rewards=all_rewards,
            gamma=data.gamma, eps=data.eps, alpha=data.alpha, policy_type=PolicyType(data.policy_type)
        )


    def save(self, filename):
        data = dict(
            transition_function_type=self.model.config.transition_function_type.value, 
            reward_function_type=self.model.config.reward_function_type.value, 
            n_states=self.model.config.n_states, 
            n_actions=self.model.config.n_actions, 
            n_rewards=len(self.model.all_rewards), 
            gamma=self.policy.gamma, 
            eps=self.policy.eps, 
            alpha=1e-3 if self.policy_type in (PolicyType.VI, PolicyType.PI) else self.policy.alpha,
            policy_type=self.policy_type.value,
            terminate_s_flags=(1 * self.terminate_s_flags).tolist(),
            transitions=self.transitions.tolist(),
            rewards=self.rewards.tolist(),
            all_rewards=self.all_rewards.tolist()
        )
        with open(filename, "w") as file:
            json.dump(data, file)


    def train_policy(self, **args):
        self.policy.fit(**args)


    def get_policy(self):
        return self.policy.get_policy()


    def play(self, n_steps=100, verbose=True, seed=42):
        observation, info = self.model.reset(seed=seed)
        if verbose:
            print(". =>", observation, None, False, False, info)

        policy = self.policy.get_policy()
        for _ in range(n_steps):
            action = policy[observation]
            observation, reward, terminated, truncated, info = self.model.step(action)
            if verbose:
                print(action, "=>", observation, reward, terminated, truncated, info)

            if terminated or truncated:
                break

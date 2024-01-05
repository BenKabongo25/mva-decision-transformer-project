# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np

from enums import MDPTransitionType, MDPRewardType, SpaceType, PolicyType
from mdp import MDP, MDPConfig
from policies import VI, PI
from td_policies import SARSA, QLearning, DoubleQLearning
from utils import terminate_s, transition, reward
from wrappers import DiscreteActionWrapper, DiscreteObservationWrapper, Range


class MDPFactory(object):

    def __init__(self, transition_function_type, reward_function_type, 
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
        self.model = MDP(config)

        self.terminate_s_flags = terminate_s(n_states, p=p)
        self.transitions = transition(config.transition_function_type, n_states, n_actions)
        all_rewards = np.arange(-n_rewards + 2, 2, 1)
        self.rewards = reward(config.reward_function_type, n_states, n_actions, 
                              self.terminate_s_flags, self.transitions, all_rewards)

        terminate_function = lambda s: self.terminate_s_flags[s]

        def transition_function(s, a, next_s):
            if transition_function_type is MDPTransitionType.S_DETERMINISTIC:
                return self.transitions[s]

            if transition_function_type is MDPTransitionType.S_PROBABILISTIC:
                probas = self.transitions[s]
                next_ss = np.arange(len(probas))
                return dict(zip(next_ss, probas))

            if transition_function_type is MDPTransitionType.SA_DETERMINISTIC:
                return self.transitions[s, a]

            if transition_function_type is MDPTransitionType.SA_PROBABILISTIC:
                probas = self.transitions[s, a]
                next_ss = np.arange(len(probas))
                return dict(zip(next_ss, probas))

            if transition_function_type is MDPTransitionType.SAS:
                return self.transitions[s, a, next_s]

            raise ValueError("Unknown transition type")

        def reward_function(s, a, next_s, r):
            if reward_function_type is MDPRewardType.S:
                return self.rewards[s]

            if reward_function_type is MDPRewardType.SA:
                return self.rewards[s, a]

            if reward_function_type is MDPRewardType.SAS:
                return self.rewards[s, a, s]

            if reward_function_type is MDPRewardType.SASR:
                i = list(all_rewards).index(r)
                return self.rewards[s, a, next_s, i]
            
            raise ValueError("Unknown reward type")
            
        observation_wrapper = DiscreteObservationWrapper(self.model, Range(n_states))
        action_wrapper = DiscreteActionWrapper(self.model, Range(n_actions))

        self.model.init(
            observation_wrapper, 
            action_wrapper, 
            transition_function, 
            reward_function, 
            terminate_function, 
            all_rewards
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


    def train_policy(self, **args):
        self.policy.fit(**args)


    def get_policy(self):
        return self.policy.get_policy()


    def play(self, n_steps=100, verbose=True):
        observation, info = self.model.reset(seed=42)
        print(". =>", observation, None, False, False, info)
        policy = self.policy.get_policy()

        for _ in range(n_steps):
            action = policy[observation]
            observation, reward, terminated, truncated, info = self.model.step(action)
            if verbose:
                print(action, "=>", observation, reward, terminated, truncated, info)

            if terminated or truncated:
                break

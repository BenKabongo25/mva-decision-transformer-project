# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np
from enums import MDPTransitionType, MDPRewardType


## Terminate functions ############################################################################

def terminate_s(n_states, p=0.1, exclude_start=True):
    states = np.random.choice([True, False], p=[p, 1-p], size=n_states)
    if exclude_start:
        states[0] = False
    if np.all(states == False):
        states[-1] = True
    return states


## Transitions functions ##########################################################################

def transition_s_deterministic(n_states):
    transitions = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        other_s = list(range(n_states))
        del other_s[s]
        transitions[s] = np.random.choice(other_s)
    return transitions


def transition_s_probabilistic(n_states, high=10):
    probas = np.random.randint(low=1, high=high+1, size=(n_states, n_states))
    probas -= 1 * np.any(probas > 1, axis=1)[:, np.newaxis]
    probas = probas / probas.sum(1)[:, np.newaxis]
    return probas


def transition_sa_deterministic(n_states, n_actions):
    return np.random.randint(low=0, high=n_states, size=(n_states, n_actions))


def transition_sa_probabilistic(n_states, n_actions, high=10):
    probas = np.random.randint(low=1, high=high+1, size=(n_states, n_actions, n_states))
    probas -= 1 * np.any(probas > 1, axis=2)[:, :, np.newaxis]
    probas = probas / probas.sum(2)[:, :, np.newaxis]
    return probas


transition_sas = transition_sa_probabilistic


def transition(transition_type, n_states, n_actions, high=10):
    if transition_type is MDPTransitionType.S_DETERMINISTIC:
        return transition_s_deterministic(n_states)
    if transition_type is MDPTransitionType.S_PROBABILISTIC:
        return transition_s_probabilistic(n_states, high=high)
    if transition_type is MDPTransitionType.SA_DETERMINISTIC:
        return transition_sa_deterministic(n_states, n_actions)
    return transition_sas(n_states, n_actions, high=high)


## Reward functions ###############################################################################

def reward_s(n_states, terminate_states, other_rewards=[0], max_reward=1):
    if len(other_rewards) == 1:
        rewards = other_rewards[0] * np.ones(n_states)
    else:
        rewards = np.random.choice(other_rewards, size=n_states)
    rewards[terminate_states] = max_reward
    return rewards
    

def reward_sa(n_states, n_actions, terminate_states, transitions, other_rewards=[0], max_reward=1):
    if len(other_rewards) == 1:
        rewards = other_rewards[0] * np.ones((n_states, n_actions))
    else:
        rewards = np.random.choice(other_rewards, size=(n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            if terminate_states[transitions[s, a]]:
                rewards[s, a] = max_reward
    return rewards


def reward_sas(n_states, n_actions, terminate_states, other_rewards=[0], max_reward=1):
    if len(other_rewards) == 1:
        rewards = other_rewards[0] * np.ones((n_states, n_actions, n_states))
    else:
        rewards = np.random.choice(other_rewards, size=(n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            for next_s in range(n_states):
                if terminate_states[next_s]:
                    rewards[s, a, next_s] = max_reward
    return rewards


def reward_sasr(n_states, n_actions, n_rewards, terminate_states, high=10):
    rewards = np.zeros((n_states, n_actions, n_states, n_rewards))
    for s in range(n_states):
        for a in range(n_actions):
            for next_s in range(n_states):
                if terminate_states[next_s]:
                    rewards[s, a, next_s, -1] = 1.
                else:
                    probas = np.random.randint(low=1, high=high + 1, size=n_rewards - 1)
                    if np.any(probas > 1):
                        probas = probas - 1
                    probas = probas / probas.sum()
                    rewards[s, a, next_s, :-1] = probas
    return rewards


def reward(reward_type, n_states, n_actions, terminate_states, transitions, all_rewards, high=10):
    all_rewards = np.sort(all_rewards)
    other_rewards = all_rewards[:-1]
    max_reward = all_rewards[-1]
    n_rewards = len(all_rewards)

    if reward_type is MDPRewardType.S:
        return reward_s(n_states, terminate_states, other_rewards, max_reward)
    if reward_type is MDPRewardType.SA:
        return reward_sa(n_states, n_actions, terminate_states, transitions, other_rewards, max_reward)
    if reward_type is MDPRewardType.SAS:
        return reward_sas(n_states, n_actions, terminate_states, other_rewards, max_reward)
    return reward_sasr(n_states, n_actions, n_rewards, terminate_states, high)

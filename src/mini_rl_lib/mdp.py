# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import enum
import gym
import gym.spaces
import numpy as np
from typing import Any, Union, Callable
from wrappers import DiscreteWrapper


class SpaceType(enum.Enum):
    DISCRETE   = 0
    CONTINUOUS = 1


class MDPTransitionType(enum.Enum):
    S_DETERMINISTIC = 0
    S_PROBABILISTIC = 1
    SA_DETERMINISTIC = 2
    SA_PROBABILISTIC = 3
    SAS = 4


class MDPRewardType(enum.Enum):
    S = 0
    SA = 1
    SAS = 2
    SASR = 3


class MDPConfig(object):
    """ MDP configuration object 
    
    :param state_space_type: type of state space: DISCRETE or CONTINUOUS
    :param action_space_type: type of action space: DISCRETE or CONTINUOUS
    :param transition_function_type: type of transition function (S, SA, SAS) and (DETERMINISTIC, PROBABILISTIC)
    :param reward_function_type: type of reward function (S, SA, SAS, SASR)
    :param n_states: number of states if known
    :param n_actions: number of actions if known
    """

    def __init__(self,
            state_space_type: SpaceType=SpaceType.DISCRETE,
            action_space_type: SpaceType=SpaceType.DISCRETE,
            transition_function_type: MDPTransitionType=MDPTransitionType.SA_DETERMINISTIC,
            reward_function_type: MDPRewardType=MDPRewardType.SA,
            n_states: int=-1,
            n_actions: int=-1
        ):
        self.state_space_type = state_space_type
        self.action_space_type = action_space_type
        self.transition_function_type = transition_function_type
        self.reward_function_type = reward_function_type
        self.n_states = n_states
        self.n_actions = n_actions


class MarkovDecisionProcess(gym.Env):
    """Base class for different Markov Decision Process types

    :param config: MDP config
    :param states: observation space | wrapper
    :param actions: action space | wrapper
    :param transition_function
    :param reward_function
    """

    def __init__(self, 
                config: MDPConfig,
                states: Union[DiscreteWrapper, gym.spaces.Space], 
                actions: Union[DiscreteWrapper, gym.spaces.Space],
                transition_function: Callable=lambda s, a, next_s: None,
                reward_function: Callable=lambda s, a, next_s, r: None,
                all_rewards: Union[list, np.ndarray]=None):

        self.config = config

        self.observation_wrapper = None
        self.action_wrapper = None

        if isinstance(states, gym.ObservationWrapper):
            self.observation_wrapper = states
            self.observation_space = states.observation_space
        elif isinstance(states, gym.spaces.Space):
            self.observation_space = states
        else:
            raise TypeError("States: Type unrecognized")

        if isinstance(actions, gym.ActionWrapper):
            self.action_wrapper = actions
            self.action_space = actions.action_space
        elif isinstance(actions, gym.spaces.Space):
            self.action_space = actions
        else:
            raise TypeError("Actions: Type unrecognized")  

        if self.config.state_space_type is SpaceType.DISCRETE:
            assert isinstance(self.observation_wrapper, DiscreteWrapper)
            self.config.n_states = self.observation_wrapper.n_values
        
        if self.config.action_space_type is SpaceType.DISCRETE:
            assert isinstance(self.action_wrapper, DiscreteWrapper)
            self.config.n_actions = self.observation_wrapper.n_values

        self._transition_function = transition_function
        self._reward_function = reward_function

        self.all_rewards = all_rewards
        if self.config.reward_function_type is MDPRewardType.SASR:
            assert self.all_rewards is not None


    def transition_function(self, s: Any, a: Any=None, next_s: Any=None) -> Union[float, dict, Any, np.ndarray]:
        """
        Transition function between states
        :param s: current state
        :param a (optional): action
        :param next_s (optional): next state 
        :return 
            transition type = S/SA and Deterministic
                next_s: next_state
            transition type = S/SA and Probabilistic
                dict: key = next_s, value = probability
            transition type = SAS
                float: probability
        """
        assert s is not None
        assert self.observation_space.contains(s)

        if self.config.transition_function_type is MDPTransitionType.S_DETERMINISTIC:
            next_s = self._transition_function(s, None, None)
            assert self.observation_space.contains(next_s)
            return next_s

        if self.config.transition_function_type is MDPTransitionType.S_PROBABILISTIC:
            next_s_probs = self._transition_function(s, None, None)
            if self.config.n_states != -1:
                assert len(next_s_probs) == self.config.n_states
            next_ss, probs = next_s_probs.items()
            for next_s in next_ss:
                assert self.observation_space.contains(next_s)
            assert np.sum(probs) == 1
            return next_s_probs

        assert a is not None
        assert self.action_space.contains(a)

        if self.config.transition_function_type is MDPTransitionType.SA_DETERMINISTIC:
            next_s = self._transition_function(s, a, None)
            assert self.observation_space.contains(next_s)
            return next_s

        if self.config.transition_function_type is MDPTransitionType.SA_PROBABILISTIC:
            next_s_probs = self._transition_function(s, a, None)
            if self.config.n_states != -1:
                assert len(next_s_probs) == self.config.n_states
            next_ss, probs = next_s_probs.items()
            for next_s in next_ss:
                assert self.observation_space.contains(next_s)
            assert np.sum(probs) == 1
            return next_s_probs

        assert next_s is not None
        assert self.observation_space.contains(next_s)

        next_s_prob = self._transition_function(s, a, next_s)
        return next_s_prob


    def reward_function(self, s: Any, a: Any=None, next_s: Any=None, r: float=None) -> float:
        """
        Reward function
        :param s: current state
        :param a (optional): action
        :param next_s (optional): next state 
        :return float
            reward type = S/SA/SAS
                reward
            reward type = SASR
                probability
        """
        assert s is not None
        assert self.observation_space.contains(s)

        if self.config.reward_function_type in (MDPRewardType.SA, MDPRewardType.SAS, MDPRewardType.SASR):
            assert a is not None
            assert self.action_space.contains(a)

        if self.config.reward_function_type in (MDPRewardType.SAS, MDPRewardType.SASR):
            assert next_s is not None
            assert self.observation_space.contains(next_s)

        if self.config.reward_function_type in (MDPRewardType.SASR,):
            assert r is not None

        r_or_prob = self._reward_function(s, a, next_s, r)
        return r_or_prob


MDP = MarkovDecisionProcess


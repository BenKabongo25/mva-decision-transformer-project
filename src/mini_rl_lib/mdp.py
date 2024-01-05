# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import gym
import gym.spaces
import numpy as np
from typing import Any, Callable, Tuple, Union

from .enums import *
from .wrappers import DiscreteWrapper


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
    :param terminate_function
    """

    def __init__(self, config: MDPConfig):
        self.config = config 
        
        
    def init(self,
            states: Union[DiscreteWrapper, gym.spaces.Space], 
            actions: Union[DiscreteWrapper, gym.spaces.Space],
            transition_function: Callable=lambda s, a, next_s: None,
            reward_function: Callable=lambda s, a, next_s, r: None,
            terminate_function: Callable=lambda s: False,
            all_rewards: Union[list, np.ndarray]=None):

        self.observation_wrapper = None
        self.action_wrapper = None
        self._current_state = 0

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
            self.config.n_actions = self.action_wrapper.n_values

        self._transition_function = transition_function
        self._reward_function = reward_function
        self._terminate_function = terminate_function

        self.all_rewards = all_rewards
        if self.config.reward_function_type is MDPRewardType.SASR:
            assert self.all_rewards is not None


    def transition_function(self, s: int, a: int=None, next_s: int=None) -> Union[int, float, dict]:
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
        assert self.observation_space.contains(s), str(s)

        if self.config.transition_function_type is MDPTransitionType.S_DETERMINISTIC:
            next_s = self._transition_function(s, None, None)
            assert self.observation_space.contains(next_s), str(next_s)
            return next_s

        if self.config.transition_function_type is MDPTransitionType.S_PROBABILISTIC:
            next_s_probs = self._transition_function(s, None, None)
            if self.config.n_states != -1:
                assert len(next_s_probs) == self.config.n_states
            next_ss, probs = list(next_s_probs.keys()), list(next_s_probs.values())
            for next_s in next_ss:
                assert self.observation_space.contains(next_s), str(next_s)
            #assert np.sum(probs) == 1
            return next_s_probs

        assert a is not None
        assert self.action_space.contains(a), str(a)

        if self.config.transition_function_type is MDPTransitionType.SA_DETERMINISTIC:
            next_s = self._transition_function(s, a, None)
            assert self.observation_space.contains(next_s), str(next_s)
            return next_s

        if self.config.transition_function_type is MDPTransitionType.SA_PROBABILISTIC:
            next_s_probs = self._transition_function(s, a, None)
            if self.config.n_states != -1:
                assert len(next_s_probs) == self.config.n_states
            next_ss, probs = list(next_s_probs.keys()), list(next_s_probs.values())
            for next_s in next_ss:
                assert self.observation_space.contains(next_s), str(next_s)
            #assert np.sum(probs) == 1
            return next_s_probs

        assert next_s is not None
        assert self.observation_space.contains(next_s), str(next_s)

        next_s_prob = self._transition_function(s, a, next_s)
        return next_s_prob


    def reward_function(self, s: int, a: int=None, next_s: int=None, r: float=None) -> float:
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
        assert self.observation_space.contains(s), str(a)

        if self.config.reward_function_type in (MDPRewardType.SA, MDPRewardType.SAS, MDPRewardType.SASR):
            assert a is not None
            assert self.action_space.contains(a), str(a)

        if self.config.reward_function_type in (MDPRewardType.SAS, MDPRewardType.SASR):
            assert next_s is not None
            assert self.observation_space.contains(next_s), str(next_s)

        if self.config.reward_function_type in (MDPRewardType.SASR,):
            assert r is not None

        r_or_prob = self._reward_function(s, a, next_s, r)
        return r_or_prob


    def reset(self, seed: int=None, options: dict=None) -> Tuple[int, dict]:
        super().reset(seed=seed, options=options)
        observation = self._current_state
        info = {}
        self._current_state = 0
        return observation, info


    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        s = self._current_state
        a = action
        next_s = None
        r = None
        info = {}
        terminated = False

        if self.config.transition_function_type in (MDPTransitionType.S_DETERMINISTIC,
                                                    MDPTransitionType.SA_DETERMINISTIC):
            next_s = self.transition_function(s, a, None)

        elif self.config.transition_function_type in (MDPTransitionType.S_PROBABILISTIC,
                                                      MDPTransitionType.SA_PROBABILISTIC):
            next_s_probs = self.transition_function(s, a, None)
            next_ss, probs = list(next_s_probs.keys()), list(next_s_probs.values())
            next_s = self.np_random.choice(next_ss, p=probs)
            
        else:
            probs = []
            for next_s in range(self.config.n_states):
                p_s = self.transition_function(s, a, next_s)
                probs.append(p_s)
            probs = np.array(probs)
            if probs.sum() == 0:
                return None, r, terminated, True, {"Error": "Probabilities sum to zero"}
            probs = probs / probs.sum()
            next_s = self.np_random.choice(np.arange(self.config.n_states), p=probs)

        if self.config.reward_function_type is not MDPRewardType.SASR:
            r = self.reward_function(s, a, next_s, None)
                
        else:
            probs = []
            for r in self.all_rewards:
                p_r = self.reward_function(s, a, next_s, r)
                probs.append(p_r)
            probs = np.array(probs)
            if probs.sum() == 0:
                return next_s, r, terminated, True, {"Error": "Probabilities sum to zero"}
            probs = probs / probs.sum()
            r = self.np_random.choice(self.all_rewards, p=probs)

        terminated = self._terminate_function(next_s)
        self._current_state = next_s
        
        return next_s, r, terminated, False, info


MDP = MarkovDecisionProcess


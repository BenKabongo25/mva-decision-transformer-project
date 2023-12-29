# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import enum
import gym
import gym.spaces
import numpy as np
from typing import Any, Union
from wrappers import DiscreteWrapper


class SpaceType(enum.Enum):
    DISCRETE   = 0
    CONTINUOUS = 1


class MDPFunctionType(enum.Enum):
    S_DETERMINISTIC = 0
    S_PROBABILISTIC = 1
    SA_DETERMINISTIC = 0
    SA_PROBABILISTIC = 1
    SAS_DETERMINISTIC = 0
    SAS_PROBABILISTIC = 1


class MDPConfig(object):
    """ MDP configuration object 
    
    :param state_space_type: type of state space: DISCRETE or CONTINUOUS
    :param action_space_type: type of action space: DISCRETE or CONTINUOUS
    :param transition_function_type: type of transition function (S, SA, SAS) and (DETERMINISTIC, PROBABILISTIC)
    :param reward_function_type: type of transition function (S, SA, SAS) and (DETERMINISTIC, PROBABILISTIC)
    :param n_states: number of states if known
    :param n_actions: number of actions if known
    """

    def __init__(self,
            state_space_type: SpaceType=SpaceType.DISCRETE,
            action_space_type: SpaceType=SpaceType.DISCRETE,
            transition_function_type: MDPFunctionType=MDPFunctionType.SA_DETERMINISTIC,
            reward_function_type: MDPFunctionType=MDPFunctionType.SA_DETERMINISTIC,
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
    """

    def __init__(self, 
                config: MDPConfig,
                states: Union[DiscreteWrapper, gym.spaces.Space], 
                actions: Union[DiscreteWrapper, gym.spaces.Space]):   
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

        self.config = config

        if self.config.state_space_type is SpaceType.DISCRETE:
            assert isinstance(self.observation_wrapper, DiscreteWrapper)
        
        if self.config.action_space_type is SpaceType.DISCRETE:
            assert isinstance(self.action_wrapper, DiscreteWrapper)


    def transition_function(self, s: Any, a: Any, next_s: Any=None) -> Union[float, np.ndarray]:
        """
        Transition function between states
        :param s: current state
        :param a: action
        :param next_s (optional): next state 
        """
        raise NotImplementedError

    def reward_function(self, s: Any, a: Any, next_s: Any=None) -> float:
        """
        Reward function
        :param s: current state
        :param a: action
        :param next_s (optional): next state 
        """
        raise NotImplementedError


MDP = MarkovDecisionProcess


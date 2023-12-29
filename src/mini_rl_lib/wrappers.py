# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import gym
import gym.spaces
import gym.wrappers
import numpy as np
from typing import Any, Union


## Range ##########################################################################################

class Range(object):

    def __init__(self, end: int, start: int=0, step: int=1):
        assert end >= start + step
        self.start = start
        self.step = step
        self.end = end
        self.values = list(range(start, end, step))
        self.n = len(self.values)


## Discrete Space #################################################################################

class DiscreteSpaceWrapper(object):

    def __init__(self, range_: Range):
        self.range = range_
        self.n_values = self.range.n

    def get(self, index: int) -> int:
        assert 0 <= index < self.n_values
        return self.range.values[index]


class DiscreteActionWrapper(gym.ActionWrapper, DiscreteSpaceWrapper):

    def __init__(self, env: gym.Env, range_: Range):
        gym.ActionWrapper.__init__(self, env)
        DiscreteSpaceWrapper.__init__(self, range_)
        self.action_space = gym.spaces.Discrete(self.n_values)

    def action(self, action: int) -> int:
        return self.get(action)


class DiscreteObservationWrapper(gym.ObservationWrapper, DiscreteSpaceWrapper):

    def __init__(self, env: gym.Env, range_: Range):
        gym.ObservationWrapper.__init__(self, env)
        DiscreteSpaceWrapper.__init__(self, range_)
        self.observation_space = gym.spaces.Discrete(self.n_values)

    def observation(self, observation: int) -> int:
        return self.get(observation)


## Multi Binary Spaces ############################################################################

class DiscreteMultiBinaryWrapper(object):

    def __init__(self, n: int):
        assert n > 0
        self.n = n
        self.pows = 2 ** np.arange(n - 1, -1, -1)
        self.n_values = 2 ** n

    def array2int(self, array: Union[list, np.ndarray]) -> int:
        if isinstance(array, (tuple, list)):
            array = np.array(array, dtype=int)
        return (array * self.pows).sum()

    def int2array(self, value: int) -> np.ndarray:
        int2 = bin(value)
        intstr = str(int2)[2:]
        intstr = "0" * (self.n - len(intstr)) + intstr
        array = np.array(list(map(int, intstr)))
        return array

    get = int2array


class DiscreteActionMultiBinaryWrapper(gym.ActionWrapper, DiscreteMultiBinaryWrapper):

    def __init__(self, env: gym.Env, n: int):
        gym.ActionWrapper.__init__(self, env)
        DiscreteMultiBinaryWrapper.__init__(self, n)
        self.action_space = gym.spaces.Discrete(self.n_values)

    def action(self, action: int) -> np.ndarray:
        return self.get(action)


class DiscreteObservationMultiBinaryWrapper(gym.ObservationWrapper, DiscreteMultiBinaryWrapper):

    def __init__(self, env: gym.Env, n: int):
        gym.ObservationWrapper.__init__(self, env)
        DiscreteMultiBinaryWrapper.__init__(self, n)
        self.observation_space = gym.spaces.Discrete(self.n_values)

    def observation(self, observation: Any) -> np.ndarray:
        return self.get(observation)


## Multi Discrete Spaces ##########################################################################

class DiscreteMultiDiscreteWrapper(object):

    def __init__(self, ns: Union[tuple, list, np.array]):
        if isinstance(ns, (tuple, list)):
            ns = np.array(ns)
        assert np.all(ns > 0)
        self.ns = ns
        self.n = len(ns)
        self.n_values = np.prod(ns)

    def array2int(self, array: Union[list, np.ndarray]) -> int:
        if isinstance(array, (tuple, list)):
            array = np.array(array, dtype=int)
        assert len(array) == len(self.ns) and np.all(array > 0)

        value = 0
        for i, a in enumerate(array):
            factor = 1
            if i + 1 < self.n:
                factor = np.prod(self.ns[i + 1:])
            value += a * factor
        return value

    def int2array(self, value: int) -> np.ndarray:
        array = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            factor = 1
            if i + 1 < self.n:
                factor = np.prod(self.ns[i + 1:])
            a = value / factor
            array[i] = a
            value = value % factor
        return array

    get = int2array


class DiscreteActionMultiDiscreteWrapper(gym.ActionWrapper, DiscreteMultiDiscreteWrapper):

    def __init__(self, env: gym.Env, ns: Union[tuple, list, np.array]):
        gym.ActionWrapper.__init__(self, env)
        DiscreteMultiDiscreteWrapper.__init__(self, ns)
        self.action_space = gym.spaces.Discrete(self.n_values)

    def action(self, action: int) -> np.ndarray:
        return self.get(action)


class DiscreteObservationDiscreteWrapper(gym.ObservationWrapper, DiscreteMultiDiscreteWrapper):

    def __init__(self, env: gym.Env, ns: Union[tuple, list, np.array]):
        gym.ObservationWrapper.__init__(self, env)
        DiscreteMultiDiscreteWrapper.__init__(self, ns)
        self.observation_space = gym.spaces.Discrete(self.n_values)

    def observation(self, observation: int) -> np.ndarray:
        return self.get(observation)
        
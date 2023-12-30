# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import mdp
import numpy as np


class MDPPolicy(object):
    
    def __init__(self, model: mdp.MDP):
        self.model = model

    def fit(self):
        raise NotImplementedError


## Deterministic MDP Policy #######################################################################


class DeterministicMDPPolicy(MDPPolicy):
    
    def __init__(self, model: mdp.MDP, gamma: float=1e-3, eps: float=1e-3):
        super().__init__(model)
        self.gamma = gamma
        self.eps = eps


class PolicyIterationDeterministicMDPPolicy(DeterministicMDPPolicy):

    def __init__(self, model: mdp.MDP, gamma: float=1e-3, eps: float=1e-3):
        super().__init__(model, gamma, eps)
        self._values = np.zeros(model.config.n_states)
        self._policy = np.zeros(model.config.n_states, dtype=int)


    def init(self, values: np.ndarray, policy: np.ndarray):
        assert len(values) == self.model.config.n_states
        assert len(policy) == self.model.config.n_states
        assert np.all(policy >= 0) and np.all(policy < self.model.config.n_actions)
        self._values = values
        self._policy = policy


    def _update_value(self, s: int, a: int) -> float:
        def get_value(s, a, next_s):
            v = 0

            if self.model.config.reward_function_type is not mdp.MDPRewardType.SASR:
                r = self.model.reward_function(s, a, next_s, None)
                v =  r + self.gamma * self._values[next_s]
                
            else:
                for r in self.model.all_rewards:
                    p_r = self.model.reward_function(s, a, next_s, r)
                    v += p_r * (r + self.gamma )

            return v

        v = 0

        if self.model.config.transition_function_type in (mdp.MDPTransitionType.S_DETERMINISTIC,
                                                          mdp.MDPTransitionType.SA_DETERMINISTIC):
            next_s = self.model.transition_function(s, a, None)
            v = get_value(s, a, next_s)

        elif self.model.config.transition_function_type in (mdp.MDPTransitionType.S_PROBABILISTIC,
                                                            mdp.MDPTransitionType.SA_PROBABILISTIC):
            next_s_probs = self.model.transition_function(s, a, None)
            for next_s, p_s in next_s_probs.items():
                v += p_s * get_value(s, a, next_s)
            
        else:
            other_s = list(range(self.model.config.n_states))
            del other_s[s]

            for next_s in other_s:
                next_s_probs = self.model.transition_function(s, a, next_s)
                for next_s, p_s in next_s_probs.items():
                    v += p_s * get_value(s, a, next_s)

        return v


    def _update_policy(self, s: int) -> int:
        values = np.zeros(self.model.config.n_actions)
        for a in range(self.model.config.n_actions):
            v = self._update_value(s, a)
            values[a] = v
        return np.argmax(values)


    def _policy_evaluation(self):
        delta = self.eps + 1
        while delta > self.eps:
            delta = 0
            for s in range(self.model.config.n_states):
                v = self._values[s]
                a = self._policy[s]
                self._values[s] = self._update_value(s, a)
                delta = max(delta, np.abs(v - self._values[s]))

    
    def _policy_improvement(self):
        policy_stable = True
        for s in range(self.model.config.n_states):
            a = self._policy[s]
            self._policy[s] = self._update_policy(s)
            if self._policy[s] != a:
                policy_stable = False
        return policy_stable


    def fit(self):
        policy_stable = False
        while not policy_stable:
            self._policy_improvement()
            policy_stable = self._policy_evaluation()



## Probabilistic MDP Policy #######################################################################

class ProbabilisticMDPPolicy(MDPPolicy):
    
    def __init__(self, model: mdp.MDP):
        super().__init__(model)



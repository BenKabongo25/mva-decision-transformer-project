# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import enum


class SpaceType(enum.Enum):
    DISCRETE   = 0
    CONTINUOUS = 1


class MDPTransitionType(enum.Enum):
    S_DETERMINISTIC  = 0
    S_PROBABILISTIC  = 1
    SA_DETERMINISTIC = 2
    SA_PROBABILISTIC = 3
    SAS              = 4


class MDPRewardType(enum.Enum):
    S    = 0
    SA   = 1
    SAS  = 2
    SASR = 3
    

class PolicyType(enum.Enum):
    VI = 0
    PI = 1
    SARSA = 2
    QLEARNING = 3
    DQLEARNING = 4
    
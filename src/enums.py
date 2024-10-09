from enum import Enum


class UnderlyingModel(Enum):
    black_scholes = 'black_scholes'


class ComputationMethod(Enum):
    analytic = 'analytic'
    monte_carlo = 'monte_carlo'
    binomial_tree = 'binomial_tree'
    finite_difference = 'finite_difference'


class ExerciseKind(Enum):
    american = 'american'
    european = 'european'
    bermudan = 'bermudan'


class TreeModel(Enum):
    crr = 'crr'
    jr = 'jr'


class Sensitivities(Enum):
    delta = 'delta'
    gamma = 'gamma'
    theta = 'theta'
    vega = 'vega'

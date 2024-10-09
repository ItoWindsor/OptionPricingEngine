from typing import Callable, Any
from src import usual_payoffs, enums


class Derivative:

    def __init__(self,
                 payoff_func: Callable,
                 exercise_kind: enums.ExerciseKind,
                 is_path_dependant: bool,
                 dic_params_derivatives: dict[str, Any]
                 ):
        self.payoff = payoff_func
        self.exercise_kind = exercise_kind
        self.is_path_dependant = is_path_dependant
        self.dic_params_derivatives = dic_params_derivatives


class EuropeanCall(Derivative):

    def __init__(self, dic_params_derivatives: dict[str, Any]):
        super().__init__(
            payoff_func=usual_payoffs.payoff_call,
            exercise_kind=enums.ExerciseKind.european,
            is_path_dependant=False,
            dic_params_derivatives=dic_params_derivatives
        )


class EuropeanPut(Derivative):
    def __init__(self, dic_params_derivatives: dict[str, Any]):
        super().__init__(
            payoff_func=usual_payoffs.payoff_put,
            exercise_kind=enums.ExerciseKind.european,
            is_path_dependant=False,
            dic_params_derivatives=dic_params_derivatives
        )

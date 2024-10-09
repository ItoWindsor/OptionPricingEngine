import numpy as np

from Derivative import Derivative
import PricingMethod
import enums
from typing import Any


class ComputationEngine:

    def compute_price(self,
                      derivative: Derivative,
                      underlying_model: enums.UnderlyingModel,
                      computation_method: enums.ComputationMethod,
                      param_computation: dict[str, Any]
                      ) -> float:
        """
        This function should compute the price of a derivative (european Call, Bermudian Put, ...) , for a given underlying model (Black-Scholes, Merton, ...)
        with a specific method (binomial tree, monte-carlo, analytic, finite difference)

        Parameters
        ----------
        derivative : Derivative
            instance of a derivative
            e.g : An european Call of strike K, end_time 1, valuation_time : 0,

        underlying_model : enums.UnderlyingModel
            simple enumeration of a model
            e.g : enums.UnderlyingModels.BlackScholes

        computation_method : enums.ComputationMethod
            simple enumeration of a model
            e.g : enums.ComputationMethod.MonteCarlo

        param_computation : dict[str, Any]
            dictionary of parameters
            The dictionary should be consistent with the computation method inserted as it will be used to initiate an instance of it
            e.g : if computation_method = 'finite_difference', the keys of the dictionary should be n_time, n_space

        Returns
        ----------
        price : float
            the price of the derivative
        """

        match computation_method:
            case enums.ComputationMethod.analytic:
                pricer_analytic = PricingMethod.PricingAnalytic()
                price = pricer_analytic.compute_price(derivative, underlying_model)
                return price

            case enums.ComputationMethod.binomial_tree:
                pricer_tree = PricingMethod.PricingBinomialTree(param_computation['n_time'], param_computation['tree_model'])
                underlying_tree = pricer_tree.generate_paths(derivative, underlying_model)
                pu, pd = pricer_tree.compute_proba_up_down(derivative, underlying_model, param_computation['tree_model'])
                derivative_tree = np.zeros(underlying_tree.shape())

                derivative_tree[:, -1] = derivative.payoff(underlying_tree[:, -1], derivative.dic_params_derivatives['K'])
                for j in range(pricer_tree.n_time - 1, -1, -1):
                    derivative_tree[:j + 1, j] = np.exp(-r * h) * (
                            pu * derivative_tree[:j + 1, j + 1] + pd * derivative_tree[1:j + 2, j + 1])

            case enums.ComputationMethod.monte_carlo:
                pricer_mc = PricingMethod.PricingMonteCarlo(
                    param_computation['n_time'],
                    param_computation['n_mc'],
                )

                match derivative.exercise_kind:
                    case enums.ExerciseKind.american:
                        simulated_paths = pricer_mc.generate_paths(derivative, underlying_model)
                    case enums.ExerciseKind.bermudan:
                        simulated_paths = pricer_mc.generate_paths(derivative, underlying_model)
                    case enums.ExerciseKind.european:
                        end_time = derivative.dic_params_derivatives['end_time']
                        valuation_time = derivative.dic_params_derivatives['valuation_time']

                        simulated_payoff = derivative.payoff()

            case enums.ComputationMethod.finite_difference:
                pricer_fd = PricingMethod.PricingFiniteDifference()

    def simulate_paths(self, computation_method: enums.UnderlyingModel) -> np.array:
        match computation_method:
            case enums.ComputationMethod.monte_carlo:
                pass
            case enums.ComputationMethod.finite_difference:
                pass
            case enums.ComputationMethod.analytic:
                pass

    def check_dic_parameters(self, object_with_dict):
        pass

    def compute_sensitivities(self,
                              computation_method: enums.UnderlyingModel,
                              sensitivities: list[enums.Sensitivities],
                              errors: dict[str, float]
                              ) -> dict[str, float]:
        pass

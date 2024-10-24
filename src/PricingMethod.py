from abc import ABC
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Union
import inspect
from src import Derivative, enums, UnderlyingModel


class PricingMethod(ABC):

    def generate_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel) -> np.array:
        pass

    def plot_generated_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel):
        pass


class PricingMonteCarlo(PricingMethod):

    def __init__(self, n_time: int, n_mc: int):
        self.n_time = n_time
        self.n_mc = n_mc

    def generate_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel) -> np.array:
        """
        This function should generate the paths for a monte-carlo simulation for a specific asset under a specific model where
        the model gives the dynamic of the price
        e.g : for a black-scholes model, dSt/St = r dt + sigma * dW_t

        Parameters
        ----------
        derivative : Derivative
            instance of a derivative
            e.g : An european Call of strike K, end_time 1, valuation_time : 0,

        model : UnderlyingModel.UnderlyingModel
            Instance of an underlying model
            e.g : UnderlyingModel.BlackScholes

        Returns
        ----------
        st : np.ndarray(np.float64)
            matrix of the simulated paths
        """

        match model:
            case UnderlyingModel.BlackScholes():
                n_underlying: int = model.dic_param_model['n_underlying']
                d: int = n_underlying * self.n_mc  # total mc paths for underlying

                r: float = model.dic_param_model['r']  # float
                q: np.ndarray = np.array(model.dic_param_model['q'])  # list of n_underlying elements
                sigma: np.ndarray = model.dic_param_model['sigma']  # list of

                valuation_time: float = derivative.dic_params_derivatives['valuation_time']
                end_time: float = derivative.dic_params_derivatives['end_time']
                s: np.ndarray = np.array(model.dic_param_model['underlying_price'])

                t = np.linspace(valuation_time, end_time, self.n_time + 1)
                h = (end_time - valuation_time) / self.n_time

                w = self.generate_brownian_paths(h, d)
                st = np.zeros(w.shape)
                if n_underlying > 1:
                    rho: np.ndarray = model.dic_param_model['rho']
                    w = self.generate_correlated_brownian(w, sigma, rho, n_underlying)
                    for k in range(self.n_mc):
                        for j in range(self.n_time + 1):
                            st[k * n_underlying:(k + 1) * n_underlying, j] = s * np.exp(
                                ((r - q) - sigma ** 2) * t[j] + w[k * n_underlying:(k + 1) * n_underlying, j])
                else:
                    st = s * np.exp((r - q - 0.5 * (sigma ** 2)) * t + sigma * w)
                return st

    def generate_terminal_distribution(self, derivative: Derivative, model: UnderlyingModel.UnderlyingModel):
        match model:
            case UnderlyingModel.BlackScholes():
                n_underlying: int = model.dic_param_model['n_underlying']
                d: int = n_underlying * self.n_mc  # total mc paths for underlying

                r: float = model.dic_param_model['r']  # float
                q: np.ndarray = np.array(model.dic_param_model['q'])  # list of n_underlying elements
                sigma: np.ndarray = model.dic_param_model['sigma']  # list of
                rho: np.ndarray = model.dic_param_model['rho']
                valuation_time: float = derivative.dic_params_derivatives['valuation_time']
                end_time: float = derivative.dic_params_derivatives['end_time']
                s: np.ndarray = np.array(model.dic_param_model['underlying_price'])

                w = np.sqrt(end_time) * np.random.normal(size=(d, 1))

                mat_cholesky = np.linalg.cholesky(rho)
                sigma_mat = np.diag(sigma)
                mat = np.dot(sigma_mat, mat_cholesky)

                for k in range(self.n_mc):
                    w[k * n_underlying:(k + 1) * n_underlying, :] = np.dot(
                        mat[:, np.newaxis],
                        w[k * n_underlying:(k + 1) * n_underlying, :]
                    ).reshape((n_underlying, self.n_time + 1))

                return w

    def compute_price(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel) -> dict[str, any]:
        valuation_time: float = derivative.dic_params_derivatives['valuation_time']
        end_time: float = derivative.dic_params_derivatives['end_time']
        r: float = model.dic_param_model['r']  # float

        st: np.ndarray = self.generate_paths(derivative, model)

        ## filter the elements of the dic to only put in input the relevant parameters
        func_args = inspect.signature(derivative.payoff).parameters
        filtered_args = {k: derivative.dic_params_derivatives[k] for k in func_args if k in derivative.dic_params_derivatives}

        payoff_arr: np.ndarray = derivative.payoff(S=st, **filtered_args)
        payoff_arr = np.exp(-r * (end_time - valuation_time)) * payoff_arr
        confidence_interval = self.compute_confidence_interval(payoff_arr)
        price = np.mean(payoff_arr)

        dic_price = {
            'price': price,
            'confidence_interval': confidence_interval
        }

        return dic_price

    def generate_brownian_paths(
            self,
            h: float,
            d: int
    ) -> np.ndarray:
        w = np.insert(np.random.normal(scale=np.sqrt(h), size=(d, self.n_time)).cumsum(axis=1), 0, 0, axis=1)
        return w

    def generate_correlated_brownian(
            self,
            paths: np.ndarray,
            sigma: np.ndarray,
            rho: np.ndarray,
            n_underlying: int
    ) -> np.ndarray:

        assert rho.shape == (n_underlying, n_underlying)

        mat_cholesky = np.linalg.cholesky(rho)
        sigma_mat = np.diag(sigma)
        mat = np.dot(sigma_mat, mat_cholesky)

        correlated_paths = np.zeros(paths.shape)

        for k in range(self.n_mc):
            for i in range(1, self.n_time + 1):
                correlated_paths[k * n_underlying:(k + 1) * n_underlying, :] = np.dot(
                    mat[:, np.newaxis],
                    paths[k * n_underlying:(k + 1) * n_underlying,:]
                ).reshape((n_underlying, self.n_time + 1))

        return correlated_paths

    def compute_confidence_interval(self, computed_payoffs: np.ndarray) -> npt.NDArray[np.float64]:
        assert len(computed_payoffs) == self.n_mc
        var_payoffs = computed_payoffs.var()

        confidence_interval = np.zeros(2, dtype=np.float64)
        confidence_interval += computed_payoffs.mean()
        confidence_interval[0] -= np.sqrt(var_payoffs / self.n_mc)
        confidence_interval[1] += np.sqrt(var_payoffs / self.n_mc)

        return confidence_interval


class PricingAnalytic(PricingMethod):

    def compute_price(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel):
        r = model.dic_param_model['r']
        q = model.dic_param_model['q']
        sigma = model.dic_param_model['sigma']
        valuation_time = derivative.dic_params_derivatives['valuation_time']
        end_time = derivative.dic_params_derivatives['end_time']
        s = model.dic_param_model['underlying_price']

        match model:
            case UnderlyingModel.BlackScholes():
                match derivative:
                    case Derivative.EuropeanPut():
                        k = derivative.dic_params_derivatives['K']

                        d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * (end_time - valuation_time)) / (
                                sigma * np.sqrt(end_time - valuation_time))
                        d2 = d1 - sigma * np.sqrt(end_time - valuation_time)

                        return k * np.exp(-r * (end_time - valuation_time)) * stats.norm.cdf(-d2) - s * np.exp(
                            -q * (end_time - valuation_time)) * stats.norm.cdf(-d1)
                    case Derivative.EuropeanCall():
                        k = derivative.dic_params_derivatives['K']

                        d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * (end_time - valuation_time)) / (
                                sigma * np.sqrt(end_time - valuation_time))
                        d2 = d1 - sigma * np.sqrt(end_time - valuation_time)

                        return s * np.exp(-q * (end_time - valuation_time)) * stats.norm.cdf(d1) - k * np.exp(
                            -r * (end_time - valuation_time)) * stats.norm.cdf(d2)

            case UnderlyingModel.Bachelier:
                pass


class PricingBinomialTree(PricingMethod):

    def __init__(self, n_time, tree_model):
        self.n_time = n_time
        self.tree_model = tree_model

    def generate_paths(self,
                       derivative: Derivative.Derivative,
                       underlying_model: UnderlyingModel.UnderlyingModel,
                       ) -> np.array:

        match underlying_model:
            case UnderlyingModel.BlackScholes():
                assert underlying_model.dic_param_model['n_underlying'] == 1
                s = underlying_model.dic_param_model['underlying_price']
                u, d = self.compute_up_down(derivative, underlying_model, self.tree_model)
                generated_paths = np.zeros((self.n_time + 1, self.n_time + 1))
                for j in range(self.n_time + 1):  ## time variable
                    for i in range(j + 1):  ## space variable
                        generated_paths[i, j] = s * (u ** (j - i)) * (d ** i)

                return generated_paths

    def plot_generated_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel):
        generated_paths = self.generate_paths(derivative, model)
        s = model.dic_param_model['underlying_price']
        sigma = model.dic_param_model['sigma']
        r = model.dic_param_model['r']
        end_time = derivative.dic_params_derivatives['end_time']

        fig, ax = plt.subplots(figsize=(12, 4))
        for j in range(self.n_time + 1):  ## loop in time
            for i in range(j + 1):  ## loop in value up/down
                if j != self.n_time:
                    ax.plot([j, j + 1], [-i, -(i + 1)], 'b')
                    ax.plot([j, j + 1], [-i, -i], 'b')
                ax.plot(j, -i, 'ro')
                ax.text(j, -i - 0.3, str(round(generated_paths[i, j], 3)), ha='center', va='center')
        plt.axis('off')
        plt.title(
            f"Evolution of the price | {self.tree_model.value} model | $S_0$ = {s} - sigma = {sigma} - r = {r} - T = {end_time} | n = {self.n_time}")

        plt.show()

    def compute_up_down(self, derivative, underlying_model, tree_model):
        valuation_time = derivative.dic_params_derivatives['valuation_time']
        end_time = derivative.dic_params_derivatives['end_time']
        r = underlying_model.r
        sigma = underlying_model.sigma

        h = (end_time - valuation_time) / self.n_time

        match tree_model:
            case enums.TreeModel.jr:
                u = np.exp((r - sigma ** 2 / 2) * h + sigma * np.sqrt(h))
                d = np.exp((r - sigma ** 2 / 2) * h - sigma * np.sqrt(h))
                return u, d
            case enums.TreeModel.crr:
                u = np.exp(sigma * np.sqrt(h))
                d = 1 / u
                return u, d

    def compute_proba_up_down(self, derivative, underlying_model, tree_model):
        valuation_time = derivative.dic_params_derivatives['valuation_time']
        end_time = derivative.dic_params_derivatives['end_time']
        r = underlying_model.r

        h = (end_time - valuation_time) / self.n_time

        match tree_model:
            case enums.TreeModel.jr:
                pu = 1 / 2
                pd = 1 / 2
                return pu, pd
            case enums.TreeModel.crr:
                u, d = self.compute_up_down(derivative, underlying_model, tree_model)
                pu = (np.exp(r * h) - d) / (u - d)
                pd = 1 - pu
                return pu, pd


class PricingFiniteDifference(PricingMethod):

    def __init__(self, n_time: int, n_space: int):
        self.n_time = n_time
        self.n_space = n_space

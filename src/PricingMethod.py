from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from src import Derivative, enums, UnderlyingModel
from typing import Union


class PricingMethod(ABC):

    def generate_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel) -> np.array:
        pass

    def plot_generated_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel):
        pass


class PricingMonteCarlo(PricingMethod):

    def __init__(self, n_time, n_mc):
        self.n_time = n_time
        self.n_mc = n_mc

    def generate_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel) -> np.array:
        match model:
            case UnderlyingModel.BlackScholes():
                d = model.dic_param_model['n_underlying'] * self.n_mc
                r = model.dic_param_model['r']
                q = model.dic_param_model['q']
                sigma = model.dic_param_model['sigma']
                valuation_time = derivative.dic_params_derivatives['valuation_time']
                end_time = derivative.dic_params_derivatives['end_time']
                s = model.dic_param_model['underlying_price']

                t = np.linspace(valuation_time, end_time, self.n_time + 1)
                h = (end_time - valuation_time) / self.n_time
                w = np.insert(np.random.normal(scale=np.sqrt(h), size=(d, self.n_time)).cumsum(axis=1), 0, 0, axis=1)

                if model.dic_param_model['n_underlying'] > 1:
                    w = self.generate_correlated_brownian(w, model.dic_param_model['rho'], model.dic_param_model['n_underlying'])
                st = s * np.exp((r - q - 0.5 * (sigma ** 2)) * t + sigma * w)

                return st

    def generate_distribution(self, derivative: Derivative, model: UnderlyingModel.UnderlyingModel):
        pass

    def compute_price(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel):
        pass

    def generate_correlated_brownian(self, paths: np.ndarray, rho: Union[float, np.ndarray], n_underlying: int) -> np.ndarray:
        correlated_paths = np.zeros(paths.shape)
        if n_underlying == 2:
            rank1 = int(paths.shape[0] / 2)
            for k in range(self.n_mc):
                correlated_paths[k*self.n_mc:(k+1)*self.n_mc, :] = np.sqrt(rho) * paths[:rank1, :] + np.sqrt(1 - rho) * paths[rank1:, :]
            return correlated_paths

        else:  # n_underlying > 2:
            mat_cholesky = np.linalg.cholesky(rho)
            for k in range(self.n_mc):
                for i in range(1, self.n_time+1):
                    correlated_paths[k*n_underlying:(k+1)*n_underlying, i] = np.dot(mat_cholesky, correlated_paths[k*n_underlying:(k+1)*n_underlying, i])
            return correlated_paths


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
                s = underlying_model.underlying_price
                u, d = self.compute_up_down(derivative, underlying_model, self.tree_model)
                generated_paths = np.zeros((self.n_time + 1, self.n_time + 1))
                for j in range(self.n_time + 1):  ## time variable
                    for i in range(j + 1):  ## space variable
                        generated_paths[i, j] = s * (u ** (j - i)) * (d ** i)

                return generated_paths

    def plot_generated_paths(self, derivative: Derivative.Derivative, model: UnderlyingModel.UnderlyingModel):
        generated_paths = self.generate_paths(derivative, model)
        s = model.underlying_price
        sigma = model.sigma
        r = model.r
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

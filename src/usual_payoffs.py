import numpy as np
from typing import Callable


def payoff_call(S: np.ndarray, K: float):
    return np.maximum(S[:, -1] - K, 0)


def payoff_put(S: np.ndarray, K: float):
    return np.maximum(K - S[:, -1], 0)


def payoff_asian_call(S: np.ndarray, K: float, mean_function: Callable):
    asset_mean: np.ndarray = mean_function(S, axis=1)
    return np.maximum(asset_mean - K, 0)


def payoff_best_of_call(S: np.ndarray, K: float, n_underlying: int, n_mc: int):
    dic_idx_assets = {}
    for k in range(n_underlying):
        dic_idx_assets[k] = np.arange(k, n_mc * n_underlying, n_mc * n_underlying)
    pass

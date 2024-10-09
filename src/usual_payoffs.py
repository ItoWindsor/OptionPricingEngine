import numpy as np

payoff_call = lambda S, K: np.max(S - K, 0)
payoff_put = lambda S, K: np.max(K - S, 0)

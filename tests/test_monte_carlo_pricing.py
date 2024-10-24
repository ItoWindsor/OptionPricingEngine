import unittest
import numpy as np
from src import PricingMethod, UnderlyingModel
from src.Derivative import EuropeanCall, EuropeanPut


class MyTestCase(unittest.TestCase):
    def test_generate_brownian_paths(self):
        n_time = 100
        n_mc = 5000
        end_time = 1

        h = end_time / n_time

        pricer_mc = PricingMethod.PricingMonteCarlo(
            n_time=n_time,
            n_mc=n_mc
        )

        brownian_paths = pricer_mc.generate_brownian_paths(h, d = n_mc)

        empirical_cov_mat = np.cov(brownian_paths.T)

        t = np.linspace(0, end_time, n_time + 1)
        s = np.linspace(0, end_time, n_time + 1)
        t_mesh, s_mesh = np.meshgrid(t, s)
        theoretical_cov_mat = np.minimum(t_mesh, s_mesh)

        max_error = np.max(np.abs(theoretical_cov_mat - empirical_cov_mat))

        self.assertAlmostEqual(max_error, 0.0, delta=0.1)

    def test_european_call_black_scholes(self):
        n_time = 1000
        n_mc = 10000

        pricer_mc = PricingMethod.PricingMonteCarlo(
            n_time=n_time,
            n_mc=n_mc
        )

        dic_param_bs1 = {
            'r': 0.02,
            'q': 0.0,
            'sigma': 0.4,
            'underlying_price': 100,
            'n_underlying': 1
        }

        dic_param_bs2 = {
            'r': 0.1,
            'q': 0.1,
            'sigma': 0.6,
            'underlying_price': 60,
            'n_underlying': 1
        }

        dic_param_bs3 = {
            'r': 0.1,
            'q': 0.01,
            'sigma': 0.2,
            'underlying_price': 100,
            'n_underlying': 1
        }

        dic_param_call1 = {
            'K': 100,
            'end_time': 1,
            'valuation_time': 0
        }

        dic_param_call2 = {
            'K': 50,
            'end_time': 2,
            'valuation_time': 0
        }

        dic_param_call3 = {
            'K': 110,
            'end_time': 1,
            'valuation_time': 0
        }
        european_call1 = EuropeanCall(dic_param_call1)
        bs_model1 = UnderlyingModel.BlackScholes(dic_param_bs1)

        european_call2 = EuropeanCall(dic_param_call2)
        bs_model2 = UnderlyingModel.BlackScholes(dic_param_bs2)

        european_call3 = EuropeanCall(dic_param_call3)
        bs_model3 = UnderlyingModel.BlackScholes(dic_param_bs3)

        mc_price1 = pricer_mc.compute_price(european_call1, bs_model1)['price']
        mc_price2 = pricer_mc.compute_price(european_call2, bs_model2)['price']
        mc_price3 = pricer_mc.compute_price(european_call3, bs_model3)['price']

        true_price_european_call1 = 16.70
        true_price_european_call2 = 19.21
        true_price_european_call3 = 7.65

        self.assertAlmostEqual(mc_price1, true_price_european_call1, delta=0.5)
        self.assertAlmostEqual(mc_price2, true_price_european_call2, delta=0.5)
        self.assertAlmostEqual(mc_price3, true_price_european_call3, delta=0.5)


if __name__ == '__main__':
    unittest.main()

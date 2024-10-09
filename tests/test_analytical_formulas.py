import unittest
from src.PricingMethod import PricingAnalytic
from src.Derivative import EuropeanCall, EuropeanPut
from src.UnderlyingModel import BlackScholes


class TestPricingAnalytic(unittest.TestCase):

    def test_european_call_black_scholes(self):
        pricer = PricingAnalytic()

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
        bs_model1 = BlackScholes(dic_param_bs1)

        european_call2 = EuropeanCall(dic_param_call2)
        bs_model2 = BlackScholes(dic_param_bs2)

        european_call3 = EuropeanCall(dic_param_call3)
        bs_model3 = BlackScholes(dic_param_bs3)

        true_price_european_call1 = 16.70
        true_price_european_call2 = 19.21
        true_price_european_call3 = 7.65

        analytic_price_european_call1 = pricer.compute_price(european_call1, bs_model1)
        analytic_price_european_call2 = pricer.compute_price(european_call2, bs_model2)
        analytic_price_european_call3 = pricer.compute_price(european_call3, bs_model3)

        self.assertAlmostEqual(analytic_price_european_call1, true_price_european_call1, delta=0.01)
        self.assertAlmostEqual(analytic_price_european_call2, true_price_european_call2, delta=0.01)
        self.assertAlmostEqual(analytic_price_european_call3, true_price_european_call3, delta=0.01)

    def test_european_put_black_scholes(self):
        pricer = PricingAnalytic()

        dic_param_bs1 = {
            'r': 0.02,
            'q': 0.0,
            'sigma': 0.4,
            'underlying_price': 100,
        }

        dic_param_bs2 = {
            'r': 0.1,
            'q': 0.1,
            'sigma': 0.6,
            'underlying_price': 60,
        }

        dic_param_bs3 = {
            'r': 0.1,
            'q': 0.01,
            'sigma': 0.2,
            'underlying_price': 100,
        }

        dic_param_put1 = {
            'K': 100,
            'end_time': 1,
            'valuation_time': 0
        }

        dic_param_put2 = {
            'K': 50,
            'end_time': 2,
            'valuation_time': 0
        }

        dic_param_put3 = {
            'K': 110,
            'end_time': 1,
            'valuation_time': 0
        }
        european_put1 = EuropeanPut(dic_param_put1)
        bs_model1 = BlackScholes(dic_param_bs1)

        european_put2 = EuropeanPut(dic_param_put2)
        bs_model2 = BlackScholes(dic_param_bs2)

        european_put3 = EuropeanPut(dic_param_put3)
        bs_model3 = BlackScholes(dic_param_bs3)

        true_price_european_put1 = 14.72
        true_price_european_put2 = 11.02
        true_price_european_put3 = 8.17

        analytic_price_european_put1 = pricer.compute_price(european_put1, bs_model1)
        analytic_price_european_put2 = pricer.compute_price(european_put2, bs_model2)
        analytic_price_european_put3 = pricer.compute_price(european_put3, bs_model3)

        self.assertAlmostEqual(analytic_price_european_put1, true_price_european_put1, delta=0.01)
        self.assertAlmostEqual(analytic_price_european_put2, true_price_european_put2, delta=0.01)
        self.assertAlmostEqual(analytic_price_european_put3, true_price_european_put3, delta=0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)

import unittest
import numpy as np
from src import PricingMethod, UnderlyingModel


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


if __name__ == '__main__':
    unittest.main()

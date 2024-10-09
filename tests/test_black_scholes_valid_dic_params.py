import unittest
import numpy as np
from typing import Any
from src.UnderlyingModel import BlackScholes


class TestBlackScholesCheckDicParam(unittest.TestCase):

    def setUp(self):
        """Set up any initial data for the tests."""
        self.valid_params_single = {
            'r': 0.05,
            'sigma': 0.2,
            'q': 0.01,
            'underlying_price': 100.0,
            'n_underlying': 1
        }

        self.valid_params_multi = {
            'r': 0.05,
            'sigma': [0.2, 0.25],
            'q': np.array([0.01, 0.02]),
            'underlying_price': [100.0, 105.0],
            'n_underlying': 2,
            'rho': np.array([[1.0, 0.5], [0.5, 1.0]])
        }

    def test_valid_single_underlying(self):
        """Test that no error is raised for valid single underlying parameters."""
        try:
            model = BlackScholes(self.valid_params_single)
        except Exception as e:
            self.fail(f"BlackScholes instantiation raised an unexpected exception: {e}")

    def test_valid_multi_underlying(self):
        """Test that no error is raised for valid multi underlying parameters."""
        try:
            model = BlackScholes(self.valid_params_multi)
        except Exception as e:
            self.fail(f"BlackScholes instantiation raised an unexpected exception: {e}")

    def test_missing_required_param(self):
        """Test that missing a required parameter raises a ValueError."""
        invalid_params = self.valid_params_single.copy()
        del invalid_params['sigma']  # Remove a required parameter

        with self.assertRaises(ValueError) as context:
            BlackScholes(invalid_params)
        self.assertIn("Missing required parameter: 'sigma'", str(context.exception))

    def test_incorrect_type_param(self):
        """Test that providing a parameter with an incorrect type raises a TypeError."""
        invalid_params = self.valid_params_single.copy()
        invalid_params['sigma'] = "incorrect_type"  # Should be a float or numeric list

        with self.assertRaises(TypeError) as context:
            BlackScholes(invalid_params)
        self.assertIn("Parameter 'sigma' should be of type", str(context.exception))

    def test_inconsistent_n_underlying_single(self):
        """Test that providing a list for a single underlying raises a ValueError."""
        invalid_params = self.valid_params_single.copy()
        invalid_params['sigma'] = [0.2]  # Should be a scalar for n_underlying = 1

        with self.assertRaises(ValueError) as context:
            BlackScholes(invalid_params)
        self.assertIn(
            "Parameter 'sigma' should be a scalar (not a list or array) when 'n_underlying' is 1",
            str(context.exception)
        )

    def test_inconsistent_n_underlying_multi(self):
        """Test that providing a scalar for multiple underlyings raises a ValueError."""
        invalid_params = self.valid_params_multi.copy()
        invalid_params['sigma'] = 0.2  # Should be a list or array for n_underlying = 2

        with self.assertRaises(ValueError) as context:
            BlackScholes(invalid_params)
        self.assertIn(
            "Parameter 'sigma' should be a list or array when 'n_underlying' is 2",
            str(context.exception)
        )

    def test_invalid_rho_shape(self):
        """Test that providing an incorrectly shaped rho raises a ValueError."""
        invalid_params = self.valid_params_multi.copy()
        invalid_params['rho'] = np.array([[1.0, 0.5]])  # Not a square matrix

        with self.assertRaises(ValueError) as context:
            BlackScholes(invalid_params)
        self.assertIn(
            "Parameter 'rho' should be a square matrix of shape (2, 2)",
            str(context.exception)
        )


if __name__ == '__main__':
    unittest.main()


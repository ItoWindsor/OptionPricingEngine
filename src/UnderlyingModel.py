from abc import ABC, abstractmethod
from typing import Union, Any
import numpy as np

minimal_expected_black_scholes_dict_params = {
    'r': Union[int,float, list, np.ndarray],
    'sigma': Union[int, float, list, np.ndarray],
    'q': Union[int,float, list, np.ndarray],
    'underlying_price': Union[int, float, list, np.ndarray],
}

single_underlying_keys = ['r', 'sigma', 'q', 'underlying_price']

additional_expected_black_scholes_dict_params = {
    'n_underlying': int,
    'rho': Union[int, float, np.ndarray]
}

full_expected_black_scholes_dict_params = {
    **minimal_expected_black_scholes_dict_params,
    **additional_expected_black_scholes_dict_params
}


class UnderlyingModel(ABC):

    def __init__(self,
                 dic_param_model: dict[str, Any]
                 ) -> None:
        self.dic_param_model = dic_param_model

    @abstractmethod
    def check_dic_param(self, dic_param_model: dict[str, Any]):
        pass


class BlackScholes(UnderlyingModel):

    def __init__(self,
                 dic_param_model: dict[str, Any]
                 ) -> None:

        self.check_dic_param(dic_param_model)
        super().__init__(dic_param_model)

    def check_dic_param(self, dic_param_model) -> None:
        # Iterate through minimal_expected_black_scholes_dict_params to validate each entry
        for param, expected_type in minimal_expected_black_scholes_dict_params.items():
            if param not in dic_param_model:
                raise ValueError(f"Missing required parameter: '{param}'")

            value = dic_param_model[param]

            # Check if the value matches the expected type
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Parameter '{param}' should be of type '{expected_type}', "
                    f"but got '{type(value).__name__}'"
                )

            # If the parameter is a list or np.array, ensure all elements are numerical
            if isinstance(value, (list, np.ndarray)):
                if not all(isinstance(v, (int, float)) for v in value):
                    raise ValueError(
                        f"All elements in '{param}' should be of type 'int' or 'float'."
                    )

        # Additional checks for n_underlying and other parameters
        n_underlying = dic_param_model.get('n_underlying', 1)
        if not isinstance(n_underlying, int) or n_underlying < 1:
            raise ValueError("Parameter 'n_underlying' should be a positive integer.")

        # Consistency checks based on n_underlying
        if n_underlying == 1:
            # Ensure that all parameters are not lists or arrays when n_underlying is 1
            for param in ['sigma', 'q', 'underlying_price']:
                value = dic_param_model[param]
                if isinstance(value, (list, np.ndarray)):
                    raise ValueError(
                        f"Parameter '{param}' should be a scalar (not a list or array) when 'n_underlying' is 1."
                    )

            # Check if 'rho' is present (it should not be for a single underlying)
            if 'rho' in dic_param_model:
                raise ValueError("Parameter 'rho' should not be provided when 'n_underlying' is 1.")

        else:
            # Ensure that sigma, q, and underlying_price are lists/arrays of length n_underlying
            for param in ['sigma', 'q', 'underlying_price']:
                value = dic_param_model[param]
                if not isinstance(value, (list, np.ndarray)):
                    raise ValueError(
                        f"Parameter '{param}' should be a list or array when 'n_underlying' is {n_underlying}."
                    )
                if len(value) != n_underlying:
                    raise ValueError(
                        f"Parameter '{param}' should have {n_underlying} elements, but got {len(value)}."
                    )

            # Check if 'rho' is a matrix of appropriate size (n_underlying x n_underlying)
            if 'rho' in dic_param_model:
                rho = dic_param_model['rho']
                if not isinstance(rho, np.ndarray):
                    raise ValueError(f"Parameter 'rho' should be a numpy array when 'n_underlying' is {n_underlying}.")
                if rho.shape != (n_underlying, n_underlying):
                    raise ValueError(
                        f"Parameter 'rho' should be a square matrix of shape ({n_underlying}, {n_underlying})."
                    )

    def __str__(self) -> str:
        return (
            f"""model_name : Black Scholes model\n"""
            f"""r : {self.dic_param_model['r']}\n"""
            f"""q : {self.dic_param_model['q']}\n"""
            f"""sigma : {self.dic_param_model['sigma']}\n"""
            f"""underlying_price : {self.dic_param_model['underlying_price']}\n"""
        )


class Bachelier(UnderlyingModel):

    def __init__(self,
                 dic_param_model: dict[str, Any]
                 ) -> None:
        super().__init__(dic_param_model)

    def check_dic_param(self, dic_param_model: dict[str, Any]):
        pass

    def __str__(self) -> str:
        return (
            f"""model_name : Bachelier model\n"""
            f"""r : {self.dic_param_model['r']}\n"""
            f"""q : {self.dic_param_model['q']}\n"""
            f"""sigma : {self.dic_param_model['sigma']}\n"""
            f"""underlying_price : {self.dic_param_model['underlying_price']}\n"""
        )

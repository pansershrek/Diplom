import tensorflow as tf


class BaseFunction():

    def set_var_list(self, var_list):
        self.x = var_list

    def _linear_transformation(self, params):
        result = 0
        x_list_len = len(self.x)
        params_len = len(params)
        for idx in range(x_list_len):
            for idy in range(params_len // x_list_len):
                result += (
                    self.x[idx] ** (params_len // x_list_len - 1 - idy) *
                    params[idy + idx * x_list_len]
                )
        return result

    def __call__(params):
        raise NotImplementedError

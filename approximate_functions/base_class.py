import tensorflow as tf


class BaseApproximateFunction():

    def set_var_list(self, var_list):
        """Save list of points to class instance

        :param var_list: list of points
        :type f: list
        """
        self.x = var_list

    def __call__(params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        raise NotImplementedError

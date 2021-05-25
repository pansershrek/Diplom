import tensorflow as tf


class BaseTargetFunction():

    x2y = None

    def set_var_list(self, var_list):
        """Save list of points to class instance

        :param var_list: list of points
        :type f: list
        """
        self.x = var_list

    def set_x2y(self, x2y):
        self.x2y = x2y

    def __call__(self):
        """Сalculate function value"""
        raise NotImplementedError

import tensorflow as tf


class BaseTargetFunction():

    def set_var_list(self, var_list):
        """Save list of points to class instance

        :param var_list: list of points
        :type f: list
        """
        self.x = var_list

    def __call__(self):
        """Сalculate function value"""
        raise NotImplementedError

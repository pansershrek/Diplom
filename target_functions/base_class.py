import tensorflow as tf


class BaseFunction():

    def set_var_list(self, var_list):
        self.x = var_list

    def __call__(self, x):
        raise NotImplementedError

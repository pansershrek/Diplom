import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class ZettlaFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [convert_variables([-0.029896, 0])]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 0.25 * x[0] + (x[0] * x[0] - 2 * x[0] + x[1] ** 2) ** 2

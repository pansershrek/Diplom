import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class BraninFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([-math.pi, 12.275]),
            convert_variables([math.pi, 2.275]),
            convert_variables([9.42478, 2.475]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            (x[1] - 5.1 * x[0]**2 / (4 * math.pi ** 2) + 5 * x[0] / math.pi - 6)**2 +
            10 * (1 - 1 / (8 * math.pi)) * tf.cos(x[0] * math.pi / 180)
            + 10
        )

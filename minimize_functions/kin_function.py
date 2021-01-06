import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class KinFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 1.39325]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            -(tf.sin((x[0] - x[1]) * math.pi / 180) ** 2) *
            (tf.sin((x[0] + x[1]) * math.pi / 180) ** 2) /
            (x[0]**2 + x[1]**2) ** 0.5
        )

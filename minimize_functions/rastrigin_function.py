import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class RastriginFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0, 0, 0, 0]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            50 + (x[0] - 10 * tf.cos(2 * math.pi * x[0] * math.pi / 180)) +
            (x[1] - 10 * tf.cos(2 * math.pi * x[1] * math.pi / 180)) +
            (x[2] - 10 * tf.cos(2 * math.pi * x[2] * math.pi / 180)) +
            (x[3] - 10 * tf.cos(2 * math.pi * x[3] * math.pi / 180)) +
            (x[4] - 10 * tf.cos(2 * math.pi * x[4] * math.pi / 180)) +
        )

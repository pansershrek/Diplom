import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class AlpineFunction(BaseFunction):

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
        return tf.math.abs(
            x[0] * tf.sin(x[0] * math.pi / 180) + 0.1 * x[0] +
            x[1] * tf.sin(x[1] * math.pi / 180) + 0.1 * x[1] +
            x[2] * tf.sin(x[2] * math.pi / 180) + 0.1 * x[2] +
            x[3] * tf.sin(x[3] * math.pi / 180) + 0.1 * x[3] +
            x[4] * tf.sin(x[4] * math.pi / 180) + 0.1 * x[4]
        )

import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class SchwefelFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables(
                [420.9687, 420.9687, 420.9687, 420.9687, 420.9687]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            418.9829 * 5 - x[0] *
            tf.sin(tf.math.abs(x[0])**0.5 * math.pi / 180)
            - x[1] * tf.sin(tf.math.abs(x[1])**0.5 * math.pi / 180)
            - x[2] * tf.sin(tf.math.abs(x[2])**0.5 * math.pi / 180)
            - x[3] * tf.sin(tf.math.abs(x[3])**0.5 * math.pi / 180)
            - x[4] * tf.sin(tf.math.abs(x[4])**0.5 * math.pi / 180)
        )

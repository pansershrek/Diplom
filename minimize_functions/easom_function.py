import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class EasomFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables(
                [math.pi, math.pi]
            ),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            -tf.cos(x[0] * math.pi / 180) *
            tf.cos(x[1] * math.pi / 180) * tf.math.exp(
                -(x[0] - math.pi)**2 - (x[1] - math.pi)**2
            )
        )

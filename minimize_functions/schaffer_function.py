import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class SchafferFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 1.253115]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            0.5 + (
                tf.cos(tf.sin(tf.abs(x[0]**2 - x[1]**2) *
                              math.pi / 180) * math.pi / 180)**2 - 0.5
            ) / (1 + 0.001 * (x[0]**2 + x[1]**2)) ** 2
        )

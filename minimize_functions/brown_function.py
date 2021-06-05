import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class BrownFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0, 0, 0]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            (x[0]**2) ** (x[1]**2 + 1) + (x[1]**2) ** (x[0]**2 + 1) +
            (x[1]**2) ** (x[2]**2 + 1) + (x[2]**2) ** (x[1]**2 + 1) +
            (x[2]**2) ** (x[3]**2 + 1) + (x[3]**2) ** (x[2]**2 + 1)
        )

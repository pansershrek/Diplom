import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class SchmittWettersFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0.78547, 0.78547, 0.78547]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            1 / (1 + (x[0] - x[1]) ** 2) +
            tf.sin(((math.pi * x[1] + x[2]) * math.pi / 2 / 180)) +
            math.e ** (((x[0] + x[1]) / x[1] - 2) ** 2)
        )

import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class HimmelblauFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        return [
            convert_variables([3, 2]),
            convert_variables([-2.80518, 3.131312]),
            convert_variables([-3.779310, -3.283186]),
            convert_variables([3.584428, -1.848126]),
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

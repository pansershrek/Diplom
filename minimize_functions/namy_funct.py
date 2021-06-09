import numpy as np
import tensorflow as tf
import math
from math import pi

from .base_class import BaseFunction

from utils.utils import convert_variables


class axis_parallel_hyper_ellipsisoid_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return sum(i * x[i - 1]**2 for i in range(1, len(x) + 1))


class beale_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([3, 0.5]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * ((x[1])**2))**2 + (2.625 - x[0] + x[0] * ((x[1])**3))**2


class branin_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([-pi, 12.275]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        a, b, c, r, s, t = [
            1, 5.1 / (4 * (pi**2)), 5 / pi, 6, 10, 1 / (8 * pi)]
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * ((x[1])**2))**2 + (2.625 - x[0] + x[0] * ((x[1])**3))**2


class bukin_n6_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([-10, 1]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 100 * tf.sqrt(tf.abs(x[1] - 0.01 * x[0]**2) + 0.01 * tf.abs(x[0] + 10))


class colville_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([1, 1, 1, 1]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 100 * ((x[0]**2 - x[1])**2) + (x[0] - 1)**2 + (x[2] - 1)**2 + 90 * (x[2]**2 - x[3])**2 + 10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8 * (x[1] - 1) * (x[3] - 1)


class zakharov_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        d = range(1, len(x) + 1)
        zakharov_b = (sum(0.5 * i * x[i - 1] for i in d))**2
        return ((x[0]**2 + x[1]**2) + zakharov_b + zakharov_b**2)


class dixon_price_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([3.0814879110195774e-31, ]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (x[0] - 1)**2 + sum(i * ((2 * (x[i])**2 - x[i - 1])**2) for i in range(1, len(x)))


class drop_wave_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0.0, 0.0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return -1 * (1 + tf.cos(12 * tf.sqrt(x[0]**2 + x[1]**2))) / (0.5 * (x[0]**2 + x[1]**2) + 2)


class Eggholder_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([512, 404.2319]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (-1) * (x[1] + 47) * tf.sin(tf.sqrt(abs(x[1] + x[0] / 2 + 47))) - x[0] * tf.sin(tf.sqrt(abs(x[0] - (x[1] + 47))))


class goldstein_price_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, -1]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        part1 = (1 + ((x[0] + x[1] + 1)**2) * (19 - 14 * x[0] + 3 *
                                               ((x[0])**2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * ((x[1])**2)))
        return part1 * (30 + ((2 * x[0] - 3 * x[1])**2) * (18 - 32 * x[0] + 12 * (x[0])**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * ((x[1])**2)))


class xinsheyang_fnc(BaseFunction):

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
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            (x[0] + x[1] + x[2] + x[3]) * tf.math.exp(
                - tf.sin(x[0]**2 * math.pi / 180) -
                tf.sin(x[1]**2 * math.pi / 180)
                - tf.sin(x[2]**2 * math.pi / 180) -
                tf.sin(x[3]**2 * math.pi / 180)
            )
        )


class himmelblau_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([3, 2]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return ((x[0])**2 + x[1] - 11)**2 + (x[0] + (x[1])**2 - 7)**2


class matyas_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 0.26 * ((x[0])**2 + (x[1])**2) - 0.48 * x[0] * x[1]


class powell_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0, 0, 0, 0, 0, 0, 0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return sum(((x[(4 * i) - 3] + 10 * x[(4 * i) - 2])**2 + 5 * (x[(4 * i) - 1] - x[4 * i])**2 + (x[(4 * i) - 2] - 2 * x[(4 * i) - 1])**4 + 10 * (x[(4 * i) - 3] - x[4 * i])**4) for i in range(0, int(len(x) / 4)))


class rastrigin_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 10 * len(x) + sum((y**2 - 10 * tf.cos(2 * pi * y) for y in x))


class rosenbrock_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([1, 1]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return sum(100 * ((x[i + 1]) - (x[i])**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))


class mccormkick_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([-0.547198, -1.5472]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return (
            tf.sin((x[0] + x[1]) * math.pi / 180) +
            (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1
        )


class schaffer2_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([0, 0]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 0.5 + ((tf.sin((x[0])**2 - x[1]**2))**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2


class schwefel_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([420.9687, 420.9687]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return 418.9829 * len(x) - sum(y * tf.sin(tf.sqrt(abs(y))) for y in x)


class trid_fnc(BaseFunction):

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return [
            convert_variables([4, 6, 6, 4]),
        ]

    @staticmethod
    #@tf.function
    def __call__(x):
        """Сalculate function value in point x

        :param x: point
        :type x: list
        """
        return sum((x[i] - 1)**2 for i in range(len(x))) - sum(x[i] * x[i - 1] for i in range(1, len(x)))

funct_dict = {
    'axis_parallel_hyper_ellipsisoid_fnc': axis_parallel_hyper_ellipsisoid_fnc,
    'beale_fnc': beale_fnc,
    'branin_fnc': branin_fnc,
    'bukin_n6_fnc': bukin_n6_fnc,
    'colville_fnc': colville_fnc,
    'zakharov_fnc': zakharov_fnc,
    'dixon_price_fnc': dixon_price_fnc,
    'drop_wave_fnc': drop_wave_fnc,
    'Eggholder_fnc': Eggholder_fnc,
    'goldstein_price_fnc': goldstein_price_fnc,
    'xinsheyang_fnc': xinsheyang_fnc,
    'himmelblau_fnc': himmelblau_fnc,
    'matyas_fnc': matyas_fnc,
    'powell_fnc': powell_fnc,
    'rastrigin_fnc': rastrigin_fnc,
    'rosenbrock_fnc': rosenbrock_fnc,
    'mccormkick_fnc': mccormkick_fnc,
    'schaffer2_fnc': schaffer2_fnc,
    'schwefel_fnc': schwefel_fnc,
    'trid_fnc': trid_fnc,
}

import tensorflow as tf
import numpy as np


def convert_variables(var_list):
    """Convert list of floats to list of tf.Variables

    :param var_list: list of floats
    :type var_list: list
    :return: list of tf.Variables
    :rtype: list
    """
    return [
        tf.Variable(float(x)) for x in var_list
    ]


def convert_variables_without_trainable(var_list):
    """Convert list of floats to list of tf.Variables without gradient flow

    :param var_list: list of floats
    :type var_list: list
    :return: list of tf.Variables without gradient flow
    :rtype: list
    """
    return [
        tf.Variable(float(x), trainable=False) for x in var_list
    ]


def convert_params(vals):
    result = []
    for x in vals:
        for y in x:
            result.append(tf.Variable(float(y)))
    return result


def get_numpy_array(args):
    """Convert list of tf.Variables to np.array with floats

    :param var_list: list of tf.Variables
    :type var_list: list
    :return: np.array with floats
    :rtype: np.array
    """
    return np.array([x.numpy() for x in args], dtype=float)


def get_delta(x_old, x_cur):
    """Get L2 norm

    :param x_old: list of tf.Variables
    :type x_old: list
    :param x_cur: list of tf.Variables
    :type x_cur: list
    :return: L2 norm
    :rtype: float
    """
    return float(np.linalg.norm(
        get_numpy_array(x_old) - get_numpy_array(x_cur)
    ))

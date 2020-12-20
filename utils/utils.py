import tensorflow as tf
import numpy as np


def convert_variables(var_list):
    return [
        tf.Variable(float(x)) for x in var_list
    ]


def convert_variables_without_trainable(var_list):
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
    return np.array([x.numpy() for x in args], dtype=float)


def get_delta(x_old, x_cur):
    return float(np.linalg.norm(
        get_numpy_array(x_old) - get_numpy_array(x_cur)
    ))

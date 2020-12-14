import tensorflow as tf
import numpy as np


def convert_variables(var_list):
    return [
        tf.Variable(float(x)) for x in var_list
    ]


def get_numpy_array(args):
    return np.array([x.numpy() for x in args], dtype=float)


def get_delta(x_old, x_cur):
    return float(np.linalg.norm(
        get_numpy_array(x_old) - get_numpy_array(x_cur)
    ))

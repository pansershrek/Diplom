import numpy as np
import tensorflow as tf
import copy

from utils.generate_data import generate_x
from utils.utils import convert_variables, convert_variables_without_trainable
from utils.generate_data import generate_set

from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)

x = X[:100]
x_validate = X[420:]

"""
approximate_options = [
    {
        "x": [], # list of points where function trains
        "x_validate": [], # list of points where function validates
        "params": [], # target function parametrs
        "loss_function": None # tf.keras.losses.MSE, loss function
        "opt": None, # optimize algo
        "eps": 0, # min distance between steps
        "max_steps": 0, # max steps
    },
]
"""

approximate_options1 = [
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([5 for x in range(79)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 30,
    },
]

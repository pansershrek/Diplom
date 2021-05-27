import numpy as np
import tensorflow as tf
import copy

from utils.generate_data import generate_x
from utils.utils import convert_variables, convert_variables_without_trainable


x = [[y, y, y] for y in range(1, 101)]
x_validate = [[y + 0.5, y + 0.5, y + 0.5] for y in range(0, 100)]

approximate_options7_3 = [
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.9),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.5, nesterov=True),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5, nesterov=True),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.5, nesterov=True),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.001),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 10,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
        "params": convert_variables([1 for x in range(19)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.001),
        "eps": 0.0001,
        "max_steps": 10,
    },
]
"""
approximate_options7_3_white_noise = [
    copy.deepcopy(x) for x in approximate_options7_3
]
for x in approximate_options7_3_white_noise:
    x["noise_type"] = "white_noise"
    x["seed"] = 42

approximate_options7_3_gaussian_noise = [
    copy.deepcopy(x) for x in approximate_options7_3
]
for x in approximate_options7_3_gaussian_noise:
    x["noise_type"] = "gaussian_noise"
    x["seed"] = 42
    x["mean"] = 1
    x["std"] = 5

approximate_options7_3_salt_and_papper_noise = [
    copy.deepcopy(x) for x in approximate_options7_3
]
for x in approximate_options7_3_salt_and_papper_noise:
    x["noise_type"] = "salt_and_papper_noise"
    x["seed"] = 42
    x["probability_threshold"] = 0.3
"""

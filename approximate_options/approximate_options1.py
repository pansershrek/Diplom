import numpy as np
import tensorflow as tf

from utils.utils import convert_variables, convert_variables_without_trainable

"""
Approximate options
"""

approximate_options1 = [
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.001),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 10000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.Adam(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.Adam(learning_rate=10),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=10),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=10, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=10, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0, 5, 0.2))],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in list(np.arange(0.1, 5, 0.02))],
        "params": convert_variables([1 for x in range(11)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=10, momentum=1),
        "eps": 0.0001,
        "max_steps": 1000,
    },

]

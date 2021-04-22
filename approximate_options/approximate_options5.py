import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston


from utils.utils import convert_variables, convert_variables_without_trainable

"""
Approximate options
"""
X, y = load_boston(return_X_y=True)

x = X[:100]
x_validate = X[420:]

approximate_options5 = [
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adam(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1, momentum=0.9),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=1,  nesterov=True),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1,  nesterov=True),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
        "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate ],
        "params": convert_variables([1 for x in range(82)]),
        "loss_function": tf.keras.losses.MSE,
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01,  nesterov=True),
        "eps": 0.0001,
        "max_steps": 100,
    },
]

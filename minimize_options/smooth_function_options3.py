import tensorflow as tf

from utils.utils import convert_variables

"""
Minimize options
"""

smooth_function_options3 = [
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.45),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.47),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.49),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.51),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.53),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.485),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.505),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.506),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=18.507),
        "eps": 0.0001,
        "max_steps": 1000,
    },
]

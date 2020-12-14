import tensorflow as tf

from utils.utils import convert_variables


zettla_function_options = [
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([1, 1]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([0, -1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([0, -1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([0, -1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([0, -1]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
]

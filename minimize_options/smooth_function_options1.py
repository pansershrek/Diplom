import tensorflow as tf

from utils.utils import convert_variables

"""
Minimize options
"""

smooth_function_options1 = [
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05, momentum=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5, momentum=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([-3, -3]),
        "opt": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5),
        "eps": 0.0001,
        "max_steps": 100,
    },
]

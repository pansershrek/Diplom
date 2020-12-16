import tensorflow as tf

from utils.utils import convert_variables


himmelblau_function_options = [
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.01),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.03),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.05),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.07),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.09),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.1),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.3),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.5),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.7),
        "eps": 0.0001,
        "max_steps": 1000,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=0.9),
        "eps": 0.0001,
        "max_steps": 1000,
    },
]

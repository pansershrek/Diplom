import tensorflow as tf

from utils.utils import convert_variables


branin_function_options = [
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=1.7998),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=1.7999),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=1.8),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adam(learning_rate=1.801),
        "eps": 0.0001,
        "max_steps": 100,
    },
]

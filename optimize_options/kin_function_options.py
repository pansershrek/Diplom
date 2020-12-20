import tensorflow as tf

from utils.utils import convert_variables


kin_function_options = [
    {
        "x": convert_variables([0, 1.4]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=10),
        "eps": 0.001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([0, 1.4]),
        "opt": tf.keras.optimizers.Adam(learning_rate=20),
        "eps": 0.001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([0, 1.4]),
        "opt": tf.keras.optimizers.Adamax(learning_rate=2),
        "eps": 0.001,
        "max_steps": 100,
    },
]

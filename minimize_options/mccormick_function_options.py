import tensorflow as tf

from utils.utils import convert_variables


mccormick_function_options = [
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.99),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=1.001),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=0.99, initial_accumulator_value=0.2),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=1, initial_accumulator_value=0.2),
        "eps": 0.0001,
        "max_steps": 100,
    },
    {
        "x": convert_variables([10, 10]),
        "opt": tf.keras.optimizers.Adagrad(learning_rate=1.001, initial_accumulator_value=0.2),
        "eps": 0.0001,
        "max_steps": 100,
    },

]

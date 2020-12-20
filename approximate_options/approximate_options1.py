import tensorflow as tf

from utils.utils import convert_params, convert_variables_without_trainable


approximate_options1 = [
    {
        "x": [convert_variables_without_trainable([x, x]) for x in range(-100, 100, 2)],
        "x_validate": [convert_variables_without_trainable([x, x]) for x in range(-101, 100, 2)],
        "params": convert_params([[1.0 for x in range(3)] for y in range(2)]),
        "loss_function": tf.keras.losses.MAE,
        "opt": tf.keras.optimizers.Adam(learning_rate=1),
        "eps": 0.0001,
        "max_steps": 100,
    },
]

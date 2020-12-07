# Install TensorFlow
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

my_dpi = 95


def compare(f, x, eps=0.0001, result_val=None, _opts=None):
    opts = {
        "SGD": tf.keras.optimizers.SGD(learning_rate=0.01),
        "SGD with momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.01),
        "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.1),
        "Adam": tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
    }
    if _opts:
        opts = _opts
    result = {}
    for name, opt in opts.items():
        x_cur = tf.Variable(x)
        x_old = tf.Variable(x)
        steps_num = 0
        while True:
            steps_num += 1
            with tf.GradientTape() as tape:
                loss_val = f(x_cur)
            grads = tape.gradient(loss_val, [x_cur])
            opt.apply_gradients(zip(grads, [x_cur]))
            if np.linalg.norm(x_old.numpy() - x_cur.numpy()) < eps:
                break
            x_old = tf.identity(x_cur)
        if result_val is not None:
            result[name] = (steps_num, np.linalg.norm(
                result_val - x_cur.numpy()))
        else:
            result[name] = steps_num
        print(f"Algorithm is {name}. Steps number is {steps_num}. Finish value is {x_cur.numpy()}")
    step_nums = []
    final_dist = []
    alg_names = []
    for alg_name, step_num in result.items():
        if result_val is not None:
            step_nums.append(step_num[0])
            final_dist.append(step_num[1])
        else:
            step_nums.append(step_num)
        alg_names.append(alg_name)
    agregate_result = pd.DataFrame(
        {'Steps number': step_nums}, index=alg_names)
    if result_val is not None:
        agregate_result['Distance'] = final_dist
    ax = agregate_result.plot.bar(
        subplots=True, figsize=(800 / my_dpi, 800 / my_dpi))
    plt.show()


@tf.function
def f(x):
    tmp = tf.math.cos(x)
    return x * x - tmp
opts = {
    "SGD": tf.keras.optimizers.SGD(),
    "SGD with momentum": tf.keras.optimizers.SGD(momentum=0.01),
    "Adagrad": tf.keras.optimizers.Adagrad(),
    "Adam": tf.keras.optimizers.Adam(),
}
compare(f, np.array(-1, dtype=np.float), 0.0001, 0, opts)


@tf.function
def f(x):
    tmp = tf.math.cos(x)
    return x * x - tmp
opts = {
    "SGD": tf.keras.optimizers.SGD(),
    "SGD with momentum": tf.keras.optimizers.SGD(momentum=0.01),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.1),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.01),
}
compare(f, np.array(-1, dtype=np.float), 0.0001, 0, opts)


@tf.function
def f(x):
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]

opts = {
    "SGD": tf.keras.optimizers.SGD(),
    "SGD with momentum": tf.keras.optimizers.SGD(momentum=0.01),
    "Adagrad": tf.keras.optimizers.Adagrad(),
    "Adam": tf.keras.optimizers.Adam(),
}
compare(f, np.array([-1, 2, 1], dtype=np.float), 0.0001,
        np.array([0, 0, 0], dtype=np.float), opts)


@tf.function
def f(x):
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]

opts = {
    "SGD": tf.keras.optimizers.SGD(),
    "SGD with momentum": tf.keras.optimizers.SGD(momentum=0.01),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=1),
    "Adam": tf.keras.optimizers.Adam(learning_rate=1),
}
compare(f, np.array([-1, 2, 1], dtype=np.float), 0.0001,
        np.array([0, 0, 0], dtype=np.float), opts)


@tf.function
def f(x):
    return 3 * x[0] * x[0] + x[0] * x[1] + 2 * x[1] * x[1] - x[0] - 4 * x[1]
opts = {
    "SGD": tf.keras.optimizers.SGD(),
    "SGD with momentum": tf.keras.optimizers.SGD(momentum=0.1),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=1),
    "Adam": tf.keras.optimizers.Adam(learning_rate=1),
}

compare(f, np.array([-1, 2], dtype=np.float), 0.0001,
        np.array([0, 1], dtype=np.float), opts)


@tf.function
def f(x):
    return 4 + (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (2.0 / 3) + x[0] + x[1] + x[2]
opts = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.1),
    "SGD with momentum": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=1),
    "Adam": tf.keras.optimizers.Adam(learning_rate=1),
}

compare(f, np.array([1, 1, 1], dtype=np.float), 0.0001, np.array(
    [-1.265625, -1.265625, -1.265625], dtype=np.float), opts)

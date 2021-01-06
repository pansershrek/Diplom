import copy
import tensorflow as tf
import numpy as np
import pandas as pd

from utils.utils import get_delta, get_numpy_array


def optimize(f, optimize_option):
    """Minimize target fuction with specific option

    :param f: optimize function
    :type f: heir from BaseFunction
    :param optimize_option: specific optimize option
    :type optimize_option: dict
    :return: result of optimization
    :rtype: dict
    """
    history = {}
    x_cur = copy.deepcopy(optimize_option["x"])
    x_old = copy.deepcopy(optimize_option["x"])
    steps_num = 0
    x_min = copy.deepcopy(optimize_option["x"])
    loss_min = f(x_min)
    max_steps = optimize_option.get("max_steps", 10**10)
    eps = optimize_option["eps"]
    result_vals = f.get_minimum()
    opt = optimize_option["opt"]
    while True:
        # Minimize cycle
        if steps_num >= max_steps:
            break
        with tf.GradientTape() as t:
            loss_val = f(x_cur)
        gradients = t.gradient(loss_val, x_cur)
        opt.apply_gradients(zip(gradients, x_cur))
        x_delta = get_delta(x_old, x_cur)
        history[steps_num] = {
            "x": get_numpy_array(x_cur), "x_old": get_numpy_array(x_old),
            "loss": loss_val.numpy(), "x_delta (L2 Norm)": x_delta,
        }

        x_old = copy.deepcopy(x_cur)
        if loss_val < loss_min:
            x_min = copy.deepcopy(x_cur)
            loss_min = loss_val
        if x_delta < eps:
            break
        steps_num += 1
    min_delta = []
    min_delta_result = float('inf')
    if result_vals is not None:
        for result_val in result_vals:
            min_delta.append(
                {
                    "min": get_numpy_array(result_val),
                    "delta": get_delta(result_val, x_min),
                }
            )
            min_delta_result = min(min_delta_result, min_delta[-1]["delta"])
    result = {
        "x_final": get_numpy_array(x_cur),
        "min_delta_list (L2 Norm)": min_delta,
        "min_delta_result": min_delta_result,
        "steps_num": steps_num,
        "history": history,
        "loss_min": loss_min.numpy(),
        "x_min": get_numpy_array(x_min),
    }
    return result

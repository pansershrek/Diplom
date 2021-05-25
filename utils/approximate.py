import copy
import tensorflow as tf
import numpy as np
import pandas as pd

from utils.utils import get_delta, get_numpy_array
from utils.generate_data import add_noise


def approximate(f, f_target, approximate_option):
    """Approximate target fuction with specific option

    :param f: approximate function
    :type f: heir from BaseApproximateFunction
    :param f_target: target function
    :type f_target: heir from BaseTargetFunction
    :param approximate_option: specific approximate option
    :type approximate_option: dict
    :return: result of approximation
    :rtype: dict
    """
    history = {}
    params_cur = copy.deepcopy(approximate_option["params"])
    params_old = copy.deepcopy(approximate_option["params"])
    steps_num = 0
    params_min = copy.deepcopy(approximate_option["params"])
    f_target_val = []
    f_val = []
    var_list = approximate_option["x"]
    var_list_validate = approximate_option["x_validate"]
    loss_function = approximate_option["loss_function"]
    opt = approximate_option["opt"]
    eps = approximate_option["eps"]
    max_steps = approximate_option.get("max_steps", 10**10)
    val = approximate_option.get("val", None)
    if val:
        f_target.val = val
    full_x = np.concatenate((np.array(var_list), np.array(var_list_validate)))
    full_y = []
    for x in full_x:
        f_target.set_var_list(x)
        full_y.append(f_target())
    full_y = np.array(full_y)
    snr = approximate_option.get("snr", None)
    mean = approximate_option.get("mean", 0)
    std = approximate_option.get("std", 1)
    probability_threshold = approximate_option.get(
        "probability_threshold", 0.5
    )
    seed = approximate_option.get("seed", 42)
    noise_type = approximate_option.get("noise_type", "")
    x2y = add_noise(
        full_x, full_y, noise_type, snr, mean,
        std, probability_threshold, seed
    )
    if noise_type:
        f_target.set_x2y(x2y)

    for x in var_list:
        f.set_var_list(x)
        f_target.set_var_list(x)
        f_val.append(f(params_cur))
        f_target_val.append(f_target())
    # Calculate nearness of target and approximate function
    loss_min = loss_function(f_target_val, f_val)

    while True:
        # Approximate cycle
        if steps_num >= max_steps:
            break
        with tf.GradientTape() as t:
            f_target_val = []
            f_val = []
            for x in var_list:
                f.set_var_list(x)
                f_target.set_var_list(x)
                f_val.append(f(params_cur))
                f_target_val.append(f_target())
            # Calculate nearness of target and approximate function
            loss_val = loss_function(f_target_val, f_val)

        gradients = t.gradient(loss_val, params_cur)
        opt.apply_gradients(zip(gradients, params_cur))
        params_delta = get_delta(params_old, params_cur)
        history[steps_num] = {
            "params": get_numpy_array(params_cur), "params_old": get_numpy_array(params_old),
            "loss": loss_val.numpy(), "params_delta (L2 Norm)": params_delta,
        }

        params_old = copy.deepcopy(params_cur)
        if loss_val < loss_min:
            params_min = copy.deepcopy(params_cur)
            loss_min = loss_val
        if params_delta < eps:
            break
        steps_num += 1

    f_target_val = []
    f_val = []
    for x in var_list_validate:
        f.set_var_list(x)
        f_target.set_var_list(x)
        f_val.append(f(params_min))
        f_target_val.append(f_target())

    loss_validate = loss_function(f_target_val, f_val)

    result = {
        "steps_num": steps_num,
        "history": history,
        "loss_min": loss_min.numpy(),
        "loss_validate": loss_validate.numpy(),
        "params_min": get_numpy_array(params_min),
    }

    return result

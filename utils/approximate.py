import copy
import tensorflow as tf
import numpy as np
import pandas as pd

from utils.utils import get_delta, get_numpy_array


def approximate(f, f_target, var_list, var_list_validate, params, loss_function, opt, eps, max_steps=10**10, **kwargs):
    history = {}
    params_cur = copy.deepcopy(params)
    params_old = copy.deepcopy(params)
    steps_num = 0
    params_min = copy.deepcopy(params)
    f_target_val = []
    f_val = []
    for x in var_list:
        f.set_var_list(x)
        f_target.set_var_list(x)
        f_val.append(f(params_cur))
        f_target_val.append(f_target())
    loss_min = loss_function(f_target_val, f_val)

    while True:
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
        "loss_validate": loss_validate,
    }

    return result

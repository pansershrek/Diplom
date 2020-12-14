import copy
import tensorflow as tf
import numpy as np
import pandas as pd

from utils.utils import get_delta, get_numpy_array


def optimize(f, x, opt, eps, result_val=None, max_steps=10**10, **kwargs):
    history = {}
    x_cur = copy.deepcopy(x)
    x_old = copy.deepcopy(x)
    steps_num = 0
    x_min = copy.deepcopy(x)
    loss_min = f(x_min)
    while True:
        if steps_num >= max_steps:
            break
        with tf.GradientTape() as t:
            loss_val = f(x_cur)
        gradients = t.gradient(loss_val, x_cur)
        opt.apply_gradients(zip(gradients, x_cur))
        x_delta = 0
        x_delta = get_delta(x_old, x_cur)
        history[steps_num] = {
            "x": get_numpy_array(x_cur), "x_old": get_numpy_array(x_old),
            "loss": loss_val.numpy(), "x_delta (L1 Norm)": x_delta,
        }
        x_old = copy.deepcopy(x_cur)
        if loss_val < loss_min:
            x_min = copy.deepcopy(x_cur)
        if x_delta < eps:
            break
        steps_num += 1
    min_delta = None
    if result_val is not None:
        min_delta = get_delta(result_val, x_min)
    result = {
        "x_final": get_numpy_array(x_cur),
        "min_delta (L1 Norm)": min_delta,
        "steps_num": steps_num,
        "history": history,
        "loss_min": loss_min.numpy(),
        "x_min": get_numpy_array(x_min),
    }
    return result

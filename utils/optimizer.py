import tensorflow as tf
import numpy as np
import pandas as pd


def optimize(f, x, optimizer, result_val=None, max_steps=10**10, **kwargs):
    history = {}
    x_cur = tf.Variable(x)
    x_old = tf.Variable(x)
    steps_num = 0
    while True:
        if steps_num >= max_steps:
            break
        with tf.GradientTape() as tape:
            loss_val = f(x_cur)
        grads = tape.gradient(loss_val, [x_cur])
        optimizer.apply_gradients(zip(grads, [x_cur]))
        x_delta = np.linalg.norm(x_old.numpy() - x_cur.numpy())
        history[steps_num] = {
            "x": x_cur, "loss": loss_val,
            "x_delta": x_delta, "x_old": x_old,
        }
        if x_delta < eps:
            break
        steps_num += 1
        x_old = tf.identity(x_cur)
    x_final = x_cur.numpy()
    final_delta = None
    if result_val is not None:
        final_delta = np.linalg.norm(result - x_final)
    result = {
        "x_final": x_final,
        "final_delta": final_delta,
        "steps_num": steps_num,
        "history": history,
    }

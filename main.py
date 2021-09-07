import argparse
import logging
import json
import tensorflow as tf
from utils.process import proccess_approximate
from utils.utils import convert_variables, convert_variables_without_trainable

from approximate_functions.approximate_function1 import ApproximateFunction1
from target_functions.target_function1 import TargetFunction1
from approximate_options.approximate_options1 import approximate_options1

import numpy as np


def approximate_example(args, f, target, opt, name=""):
    """Solve the approximate problem

    :param args: command line arguments
    :type args: argparse.Namespace
    """
    result = proccess_approximate(
        f(), target(), opt
    )
    res = []
    with open(args.ans, "a") as file:
        for x in result.values():
            x["result"].pop("history")
            new_result = {
                "Name": f"{name}",
                "Val loss": str(x["result"]["loss_validate"]),
                "Train loss": str(x["result"]["loss_min"]),
            }
            res.append(new_result)
        print(json.dumps(res), file=file, flush=True)
    return res


def main():
    tf.debugging.set_log_device_placement(True)
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ans', type=str, default="ans",
        help="Pathname to file with full answer"
    )
    parser.add_argument(
        '--short-ans', type=str, default="short_ans",
        help="Pathname to file with short answer"
    )
    args = parser.parse_args()
    approximate_example(
        args, ApproximateFunction1, TargetFunction1,
        approximate_options1, f"Function"
    )


if __name__ == "__main__":
    main()

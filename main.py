import argparse
import logging
import json
import tensorflow as tf
import math
from utils.process import process_optimize, proccess_approximate
from utils.utils import rmse, L_inf

from minimize_functions.smooth_function1 import SmothFunction1
from minimize_options.smooth_function_options1 import smooth_function_options1

from minimize_functions.smooth_function2 import SmothFunction2
from minimize_options.smooth_function_options2 import smooth_function_options2

from minimize_functions.smooth_function3 import SmothFunction3
from minimize_options.smooth_function_options3 import smooth_function_options3

from minimize_functions.matyas_function import MatyasFunction
from minimize_options.matyas_function_options import matyas_function_options

from minimize_functions.zettla_function import ZettlaFunction
from minimize_options.zettla_function_options import zettla_function_options

from minimize_functions.himmelblau_function import HimmelblauFunction
from minimize_options.himmelblau_function_options import himmelblau_function_options

from minimize_functions.branin_function import BraninFunction
from minimize_options.branin_function_options import branin_function_options

from minimize_functions.mccormick_function import McCormickFunction
from minimize_options.mccormick_function_options import mccormick_function_options

from minimize_functions.kin_function import KinFunction
from minimize_options.kin_function_options import kin_function_options

from minimize_functions.schmitt_wetters_function import SchmittWettersFunction
from minimize_options.schmitt_wetters_function_options import schmitt_wetters_function_options

from approximate_functions.approximate_function1 import ApproximateFunction1
from target_functions.target_function1 import TargetFunction1
from approximate_options.approximate_options1 import approximate_options1

from approximate_functions.approximate_function2 import ApproximateFunction2
from target_functions.target_function2 import TargetFunction2
from approximate_options.approximate_options2 import approximate_options2

from approximate_functions.approximate_function3 import ApproximateFunction3
from target_functions.target_function3 import TargetFunction3
from approximate_options.approximate_options3 import approximate_options3

from approximate_functions.approximate_function4 import ApproximateFunction4
from target_functions.target_function4 import TargetFunction4
from approximate_options.approximate_options4 import approximate_options4

from approximate_functions.approximate_function5 import ApproximateFunction5
from target_functions.target_function5 import TargetFunction5
from approximate_options.approximate_options5 import approximate_options5

from approximate_functions.approximate_function6 import ApproximateFunction6
from target_functions.target_function6 import TargetFunction6
from approximate_options.approximate_options6 import approximate_options6

from approximate_functions.approximate_function7_1 import ApproximateFunction7_1
from approximate_functions.approximate_function7_2 import ApproximateFunction7_2
from approximate_functions.approximate_function7_3 import ApproximateFunction7_3
from target_functions.target_function7 import TargetFunction7, TargetFunction7_1
from approximate_options.approximate_options7_1 import approximate_options7_1, approximate_options7_1_white_noise
from approximate_options.approximate_options7_2 import approximate_options7_2, approximate_options7_2_white_noise
from approximate_options.approximate_options7_3 import approximate_options7_3, approximate_options7_3_white_noise

from approximate_functions.approximate_function8_1 import ApproximateFunction8_1
from approximate_functions.approximate_function8_2 import ApproximateFunction8_2
from approximate_functions.approximate_function8_3 import ApproximateFunction8_3
from target_functions.target_function8 import TargetFunction8, TargetFunction8_2
from approximate_options.approximate_options8_1 import approximate_options8_1, approximate_options8_1_white_noise
from approximate_options.approximate_options8_2 import approximate_options8_2, approximate_options8_2_white_noise
from approximate_options.approximate_options8_3 import approximate_options8_3, approximate_options8_3_white_noise


from approximate_functions.approximate_function9_1 import ApproximateFunction9_1
from approximate_functions.approximate_function9_2 import ApproximateFunction9_2
from approximate_functions.approximate_function9_3 import ApproximateFunction9_3
from target_functions.target_function9 import TargetFunction9, TargetFunction9_1
from approximate_options.approximate_options9_1 import approximate_options9_1, approximate_options9_1_white_noise
from approximate_options.approximate_options9_2 import approximate_options9_2, approximate_options9_2_white_noise
from approximate_options.approximate_options9_3 import approximate_options9_3, approximate_options9_3_white_noise


from utils.utils import convert_variables

from minimize_functions.xin_she_yang_function import XinSheYangFunction
from minimize_functions.wolfe_function import WolfeFunction
from minimize_functions.alpine_function import AlpineFunction
from minimize_functions.schaffer_function import SchafferFunction
from minimize_functions.brown_function import BrownFunction
from minimize_functions.schwefel_function import SchwefelFunction
from minimize_functions.easom_function import EasomFunction
from minimize_functions.brent_function import BrentFunction

from utils.utils import convert_variables, convert_variables_without_trainable


from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes

from minimize_functions.namy_funct import funct_dict

import numpy as np


def get_metods(alphas):
    methods = []
    for alpha in alphas:
        methods.append(tf.keras.optimizers.SGD(learning_rate=alpha))
    for alpha in alphas:
        methods.append(tf.keras.optimizers.SGD(
            learning_rate=alpha, momentum=0.1))
    for alpha in alphas:
        methods.append(tf.keras.optimizers.SGD(
            learning_rate=alpha, momentum=0.5))
    for alpha in alphas:
        methods.append(tf.keras.optimizers.SGD(
            learning_rate=alpha, momentum=0.9))
    for alpha in alphas:
        methods.append(tf.keras.optimizers.SGD(
            learning_rate=alpha, momentum=0.5, nesterov=True))
    for alpha in alphas:
        methods.append(tf.keras.optimizers.Adagrad(learning_rate=alpha))
    for alpha in alphas:
        methods.append(tf.keras.optimizers.Adam(learning_rate=alpha))
    return methods


def get_approximate_options(n, m, methods, losses):
    #X, y = load_boston(return_X_y=True)

    #x = X[:100]
    #x_validate = X[420:]
    x = [[y] * m for y in range(1, 25)]
    x_validate = [[y + 0.5] * m for y in range(0, 25)]
    #x = [[y] * m for y in range(1, 101)]
    #x_validate = [[y + 0.5] * m for y in range(0, 100)]
    #X, y = load_diabetes(return_X_y=True)

    #x = X[:50]
    #x_validate = X[400:]
    options = []
    for method in methods:
        options.append({
            "x": [convert_variables_without_trainable(x_tmp) for x_tmp in x],
            "x_validate": [convert_variables_without_trainable(x_tmp) for x_tmp in x_validate],
            "params": convert_variables([5 for x in range(n)]),
            "loss_function": losses,
            "opt": method,
            "eps": 0.0001,
            "max_steps": 1,
        })
    return options


def minimize_example(args, f, opt):
    """Solve the minimization problem

    :param args: command line arguments
    :type args: argparse.Namespace
    """
    result = process_optimize(
        f(), opt
    )
    step = 1
    res = []
    with open(args.ans, "w") as file:
        print(result, file=file)
    with open(args.short_ans, "w") as file:
        for x in result.values():
            x["result"].pop("history")
            print(
                f'X value: {x["result"]["x_min"]}, ' +
                f'distance to min: {x["result"]["min_delta_result"]}, ' +
                f'N steps: {x["result"]["steps_num"]}'
            )
            res.append(f'X value: {x["result"]["x_min"]}, ' + f'distance to min: {x["result"]["min_delta_result"]}, ' + f'N steps: {x["result"]["steps_num"]}')
            # if step % 21 == 0:
            #    print("##################################")
            step += 1
        print(result, file=file)
    return res

# metrix = [
#    "SGD(1)", "SGD(0.1)", "SGD(0.01)", "SGD(0.001)",
#    "SGDM(1,0.1)", "SGDM(0.1,0.1)", "SGDM(0.01,0.1)", "SGDM(0.001,0.1)",
#    "SGDM(1,0.5)", "SGDM(0.1,0.5)", "SGDM(0.01,0.5)", "SGDM(0.001,0.5)",
#    "SGDM(1,0.9)", "SGDM(0.1,0.9)", "SGDM(0.01,0.9)", "SGDM(0.001,0.9)",
#    "Nesterov(1,0.5)", "Nesterov(0.1,0.5)", "Nesterov(0.01,0.5)", "Nesterov(0.001,0.5)",
#    "Adagrad(1)", "Adagrad(0.1)", "Adagrad(0.01)", "Adagrad(0.001)",
#    "Adam(1)", "Adam(0.1)", "Adam(0.01)", "Adam(0.001)",
#]

metrix = [
    "SGD(1000)", "SGD(100)", "SGD(10)", "SGD(1)", "SGD(0.1)", "SGD(0.01)", "SGD(0.001)", "SGD(0.0001)",
    "SGDM(1000,0.1)", "SGDM(100,0.1)", "SGDM(10,0.1)", "SGDM(1,0.1)", "SGDM(0.1,0.1)", "SGDM(0.01,0.1)", "SGDM(0.001,0.1)", "SGDM(0.0001,0.1)",
    "SGDM(1000,0.5)", "SGDM(100,0.5)", "SGDM(10,0.5)", "SGDM(1,0.5)", "SGDM(0.1,0.5)", "SGDM(0.01,0.5)", "SGDM(0.001,0.5)", "SGDM(0.0001,0.5)",
    "SGDM(1000,0.9)", "SGDM(100,0.9)", "SGDM(10,0.9)", "SGDM(1,0.9)", "SGDM(0.1,0.9)", "SGDM(0.01,0.9)", "SGDM(0.001,0.9)", "SGDM(0.0001,0.9)",
    "Nesterov(1000,0.5)", "Nesterov(100,0.5)", "Nesterov(10,0.5)", "Nesterov(1,0.5)", "Nesterov(0.1,0.5)", "Nesterov(0.01,0.5)", "Nesterov(0.001,0.5)", "Nesterov(0.0001,0.5)",
    "Adagrad(1000)", "Adagrad(100)", "Adagrad(10)", "Adagrad(1)", "Adagrad(0.1)", "Adagrad(0.01)", "Adagrad(0.001)", "Adagrad(0.0001)",
    "Adam(1000)", "Adam(100)", "Adam(10)", "Adam(1)", "Adam(0.1)", "Adam(0.01)", "Adam(0.001)", "Adam(0.0001)",
]


def approximate_example(args, f, target, opt, name=""):
    """Solve the approximate problem

    :param args: command line arguments
    :type args: argparse.Namespace
    """
    result = proccess_approximate(
        f(), target(), opt
    )
    with open(args.ans, "a") as file:
        for x, y in zip(result.values(), metrix):
            x["result"].pop("history")
            new_result = {
                "Name": name,
                "Val loss": str(x["result"]["loss_validate"]),
                "Train loss": str(x["result"]["loss_min"]),
                "metrix": y,
            }
            print(json.dumps(new_result), flush=True)
            print(json.dumps(new_result), file=file, flush=True)


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
    """
    alphas = [0.0001 * (10**0.5)**x for x in range(15)]
    methods = get_metods(alphas)

    res = {}
    for fname, f in funct_dict.items():
        x_shape = len(f.get_minimum()[0])
        x_min = np.array([float(x) for x in f.get_minimum()[0]])
        x_range = [
            200 * np.random.random_sample(x_shape) + (x_min - 100) for y in range(20)
        ]
        for x_cur in x_range:
            option = []
            for method in methods:
                option.append(
                    {
                        "x": convert_variables([x for x in x_cur]),
                        "opt": method,
                        "eps": 0.0001,
                        "max_steps": 100,
                    },
                )
            res[f"{fname}|{x_cur}"] = minimize_example(args, f, option)
    import sys
    print(json.dumps(res), file=sys.stderr, flush=True)
    """
    """
    with open("asdad.py", "r") as f:
        fff = json.load(f)
    for x in fff:
        exec(fff[x]['code'])
    fff["axis_parallel_hyper_ellipsisoid_fnc"]
    x = eval("axis_parallel_hyper_ellipsisoid_fnc")

    alphas = [x / 1000.0 for x in range(1, 21)]
    methods = get_metods(alphas)
    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([-2, 2]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 10,
            },
        )
    print("axis_parallel_hyper_ellipsisoid_fnc")
    minimize_example(args, eval("axis_parallel_hyper_ellipsisoid_fnc"), option)
    """
    alphas = [1000.0 / 10**x for x in range(8)]
    methods = get_metods(alphas)

    losses = [
        [tf.keras.losses.MAE, "L_1"],
        [rmse, "L_2"],
        [L_inf, "L_inf"],
    ]

    for loss, loss_name in losses:
        for p in [[4, 10, 3]]:
            ApproximateFunction7_1.P = p[0]
            ApproximateFunction7_2.P = p[1]
            ApproximateFunction7_3.P = p[2]
            a7_1 = ApproximateFunction7_1
            a7_2 = ApproximateFunction7_2
            a7_3 = ApproximateFunction7_3
            all_approximate_options7 = [
                [[get_approximate_options(
                    3 * (p[0] - 1) + 1, 3, methods, loss), "WithoutNoise"], a7_1, "Polinom"],
                [[get_approximate_options(
                    3 * (p[1] * 2) + 1, 3, methods, loss), "WithoutNoise"], a7_2, "Furie"],
                [[get_approximate_options(
                    3 * (p[2] * 2) + 1, 3, methods, loss), "WithoutNoise"], a7_3, "Exp"],
            ]
            for all_options, approximate_function, approximate_function_name in all_approximate_options7:
                option, option_name = all_options
                approximate_example(
                    args, approximate_function, TargetFunction7_1,
                    option, f"Smoth_{loss_name}_{approximate_function_name}_{option_name}"
                )
    """
    approximate_options71 = [
        #[approximate_options7_1, "WithoutNoise"],
        [approximate_options7_1_white_noise, "WhiteNoise"],
        #[approximate_options7_1_gaussian_noise, "GaussianNoise"],
        #[approximate_options7_1_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options72 = [
        #[approximate_options7_2, "WithoutNoise"],
        [approximate_options7_2_white_noise, "WhiteNoise"],
        #[approximate_options7_2_gaussian_noise, "GaussianNoise"],
        #[approximate_options7_2_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options73 = [
        #[approximate_options7_3, "WithoutNoise"],
        [approximate_options7_3_white_noise, "WhiteNoise"],
        #[approximate_options7_3_gaussian_noise, "GaussianNoise"],
        #[approximate_options7_3_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    all_approximate_options7 = [
        [approximate_options71, ApproximateFunction7_1, "Polinom"],
        [approximate_options72, ApproximateFunction7_2, "Furie"],
        [approximate_options73, ApproximateFunction7_3, "Exp"],
    ]
    for loss, loss_name in losses:
        for all_options, approximate_function, approximate_function_name in all_approximate_options7:
            for option, option_name in all_options:
                for snr in all_snr:
                    for x in option:
                        x["loss_function"] = loss
                        x["snr"] = snr
                    approximate_example(
                        args, approximate_function, TargetFunction7,
                        option, f"Smoth_{snr}_{loss_name}_{approximate_function_name}_{option_name}"
                    )
    """
    """
    approximate_options81 = [
        #[approximate_options8_1, "WithoutNoise"],
        [approximate_options8_1_white_noise, "WhiteNoise"],
        #[approximate_options8_1_gaussian_noise, "GaussianNoise"],
        #[approximate_options8_1_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options82 = [
        #[approximate_options8_2, "WithoutNoise"],
        [approximate_options8_2_white_noise, "WhiteNoise"],
        #[approximate_options8_2_gaussian_noise, "GaussianNoise"],
        #[approximate_options8_2_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options83 = [
        #[approximate_options8_3, "WithoutNoise"],
        [approximate_options8_3_white_noise, "WhiteNoise"],
        #[approximate_options8_3_gaussian_noise, "GaussianNoise"],
        #[approximate_options8_3_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    all_approximate_options8 = [
        [approximate_options81, ApproximateFunction8_1, "Polinom"],
        [approximate_options82, ApproximateFunction8_2, "Furie"],
        [approximate_options83, ApproximateFunction8_3, "Exp"],
    ]

    for loss, loss_name in losses:
        for all_options, approximate_function, approximate_function_name in all_approximate_options8:
            for option, option_name in all_options:
                for snr in all_snr:
                    for x in option:
                        x["loss_function"] = loss
                        x["snr"] = snr
                    approximate_example(
                        args, approximate_function, TargetFunction8,
                        option, f"Continuous_{snr}_{loss_name}_{approximate_function_name}_{option_name}"
                    )
    approximate_options91 = [
        #[approximate_options9_1, "WithoutNoise"],
        [approximate_options9_1_white_noise, "WhiteNoise"],
        #[approximate_options9_1_gaussian_noise, "GaussianNoise"],
        #[approximate_options9_1_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options92 = [
        #[approximate_options9_2, "WithoutNoise"],
        [approximate_options9_2_white_noise, "WhiteNoise"],
        #[approximate_options9_2_gaussian_noise, "GaussianNoise"],
        #[approximate_options9_2_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options93 = [
        #[approximate_options9_3, "WithoutNoise"],
        [approximate_options9_3_white_noise, "WhiteNoise"],
        #[approximate_options9_3_gaussian_noise, "GaussianNoise"],
        #[approximate_options9_3_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    all_approximate_options9 = [
        [approximate_options91, ApproximateFunction9_1, "Polinom"],
        [approximate_options92, ApproximateFunction9_2, "Furie"],
        [approximate_options93, ApproximateFunction9_3, "Exp"],
    ]

    for loss, loss_name in losses:
        for all_options, approximate_function, approximate_function_name in all_approximate_options9:
            for option, option_name in all_options:
                for snr in all_snr:
                    for x in option:
                        x["loss_function"] = loss
                        x["snr"] = snr
                    approximate_example(
                        args, approximate_function, TargetFunction9,
                        option, f"Discontinuous_{snr}_{loss_name}_{approximate_function_name}_{option_name}"
                    )
    """
    """
    alphas = [x / 1000.0 for x in range(1, 21)]
    print(len(alphas))
    print(alphas)
    methods = get_metods(alphas)
    option = []
    print(len(methods))
    return 0
    for method in methods:
        option.append(
            {
                "x": convert_variables([-2, 2, 2]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("SchmittWettersFunction")
    minimize_example(args, SchmittWettersFunction, option)
    print("\n\n", "!" * 30, "\n\n")
    """
    """
    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([5, 5]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("McCormickFunction")
    minimize_example(args, McCormickFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([math.pi, math.pi, math.pi, math.pi]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("XinSheYangFunction")
    minimize_example(args, XinSheYangFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([1, 1, 1]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("WolfeFunction")
    minimize_example(args, WolfeFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([1, 1, 1, 1, 1]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("AlpineFunction")
    minimize_example(args, AlpineFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([10, 10]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("SchafferFunction")
    minimize_example(args, SchafferFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([2, 2, 2, 2]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("BrownFunction")
    minimize_example(args, BrownFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([350, 350, 350, 350, 350]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("SchwefelFunction")
    minimize_example(args, SchwefelFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([1, 1]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("EasomFunction")
    minimize_example(args, EasomFunction, option)
    print("\n\n", "!" * 30, "\n\n")

    option = []
    for method in methods:
        option.append(
            {
                "x": convert_variables([3, 3]),
                "opt": method,
                "eps": 0.0001,
                "max_steps": 100,
            },
        )
    print("BrentFunction")
    minimize_example(args, BrentFunction, option)
    print("\n\n", "!" * 30, "\n\n")
    """
    #approximate_example(args, ApproximateFunction5,TargetFunction5, approximate_options5, "Approx")

if __name__ == "__main__":
    main()

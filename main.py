import argparse
import logging
import json
import tensorflow as tf

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
from target_functions.target_function7 import TargetFunction7
from approximate_options.approximate_options7_1 import approximate_options7_1
from approximate_options.approximate_options7_2 import approximate_options7_2
from approximate_options.approximate_options7_3 import approximate_options7_3

from approximate_functions.approximate_function8_1 import ApproximateFunction8_1
from approximate_functions.approximate_function8_2 import ApproximateFunction8_2
from approximate_functions.approximate_function8_3 import ApproximateFunction8_3
from target_functions.target_function8 import TargetFunction8
from approximate_options.approximate_options8_1 import approximate_options8_1
from approximate_options.approximate_options8_2 import approximate_options8_2
from approximate_options.approximate_options8_3 import approximate_options8_3


from approximate_functions.approximate_function9_1 import ApproximateFunction9_1
from approximate_functions.approximate_function9_2 import ApproximateFunction9_2
from approximate_functions.approximate_function9_3 import ApproximateFunction9_3
from target_functions.target_function9 import TargetFunction9
from approximate_options.approximate_options9_1 import approximate_options9_1
from approximate_options.approximate_options9_2 import approximate_options9_2
from approximate_options.approximate_options9_3 import approximate_options9_3


def minimize_example(args, f, opt):
    """Solve the minimization problem

    :param args: command line arguments
    :type args: argparse.Namespace
    """
    result = process_optimize(
        f(), opt
    )
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
        print(result, file=file)

metrix = [
    "SGD(1)", "SGD(0.1)", "SGD(0.01)", "SGD(0.001)",
    "SGDM(1,0.1)", "SGDM(0.1,0.1)", "SGDM(0.01,0.1)", "SGDM(0.001,0.1)",
    "SGDM(1,0.5)", "SGDM(0.1,0.5)", "SGDM(0.01,0.5)", "SGDM(0.001,0.5)",
    "SGDM(1,0.9)", "SGDM(0.1,0.9)", "SGDM(0.01,0.9)", "SGDM(0.001,0.9)",
    "Nesterov(1,0.5)", "Nesterov(0.1,0.5)", "Nesterov(0.01,0.5)", "Nesterov(0.001,0.5)",
    "Adagrad(1)", "Adagrad(0.1)", "Adagrad(0.01)", "Adagrad(0.001)",
    "Adam(1)", "Adam(0.1)", "Adam(0.01)", "Adam(0.001)",
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
                "metix": y,
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
    all_snr = [None]
    losses = [
        [tf.keras.losses.MAE, "L_1"],
        [rmse, "L_2"],
        [L_inf, "L_inf"],
    ]
    """
    approximate_options71 = [
        #[approximate_options7_1, "WithoutNoise"],
        #[approximate_options7_1_white_noise, "WhiteNoise"],
        #[approximate_options7_1_gaussian_noise, "GaussianNoise"],
        #[approximate_options7_1_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options72 = [
        [approximate_options7_2, "WithoutNoise"],
        #[approximate_options7_2_white_noise, "WhiteNoise"],
        #[approximate_options7_2_gaussian_noise, "GaussianNoise"],
        #[approximate_options7_2_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options73 = [
        [approximate_options7_3, "WithoutNoise"],
        #[approximate_options7_3_white_noise, "WhiteNoise"],
        #[approximate_options7_3_gaussian_noise, "GaussianNoise"],
        #[approximate_options7_3_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    all_approximate_options7 = [
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
    approximate_options81 = [
        [approximate_options8_1, "WithoutNoise"],
        #[approximate_options8_1_white_noise, "WhiteNoise"],
        #[approximate_options8_1_gaussian_noise, "GaussianNoise"],
        #[approximate_options8_1_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options82 = [
        [approximate_options8_2, "WithoutNoise"],
        #[approximate_options8_2_white_noise, "WhiteNoise"],
        #[approximate_options8_2_gaussian_noise, "GaussianNoise"],
        #[approximate_options8_2_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options83 = [
        [approximate_options8_3, "WithoutNoise"],
        #[approximate_options8_3_white_noise, "WhiteNoise"],
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
"""
    approximate_options91 = [
        [approximate_options9_1, "WithoutNoise"],
        #[approximate_options9_1_white_noise, "WhiteNoise"],
        #[approximate_options9_1_gaussian_noise, "GaussianNoise"],
        #[approximate_options9_1_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options92 = [
        [approximate_options9_2, "WithoutNoise"],
        #[approximate_options9_2_white_noise, "WhiteNoise"],
        #[approximate_options9_2_gaussian_noise, "GaussianNoise"],
        #[approximate_options9_2_salt_and_papper_noise, "SaltAndPapperNoise"],
    ]
    approximate_options93 = [
        [approximate_options9_3, "WithoutNoise"],
        #[approximate_options9_3_white_noise, "WhiteNoise"],
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
    # minimize_example(args)

if __name__ == "__main__":
    main()

import argparse
import logging
import json

from utils.process import process_optimize, proccess_approximate

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


def minimize_example(args):
    """Solve the minimization problem

    :param args: command line arguments
    :type args: argparse.Namespace
    """
    result = process_optimize(
        SmothFunction1(), smooth_function_options1
    )
    with open(args.ans, "w") as file:
        print(result, file=file)
    with open(args.short_ans, "w") as file:
        for x in result.values():
            x["result"].pop("history")
            print(
                f'X value: {x["result"]["x_min"]}, ' +
                f'distance to min: {x["result"]["min_delta_result"]}'
            )
        print(result, file=file)


def approximate_example(args):
    """Solve the approximate problem

    :param args: command line arguments
    :type args: argparse.Namespace
    """
    result = proccess_approximate(
        ApproximateFunction4(), TargetFunction4(), approximate_options4
    )
    with open(args.ans, "w") as file:
        print(result, file=file)
    with open(args.short_ans, "w") as file:
        for x in result.values():
            x["result"].pop("history")
            print(
                f'Val loss: {x["result"]["loss_validate"]}, '
                f'Train loss: {x["result"]["loss_min"]},\n'
                f'params_min: {x["result"]["params_min"]}\n'
            )
        print(result, file=file)


def main():
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
    approximate_example(args)
    minimize_example(args)

if __name__ == "__main__":
    main()

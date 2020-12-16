import argparse
import logging
import json

from utils.process import process

from functions.smooth_function1 import SmothFunction1
from optimize_options.smooth_function_options1 import smooth_function_options1

from functions.smooth_function2 import SmothFunction2
from optimize_options.smooth_function_options2 import smooth_function_options2

from functions.matyas_function import MatyasFunction
from optimize_options.matyas_function_options import matyas_function_options

from functions.zettla_function import ZettlaFunction
from optimize_options.zettla_function_options import zettla_function_options

from functions.himmelblau_function import HimmelblauFunction
from optimize_options.himmelblau_function_options import himmelblau_function_options

from functions.branin_function import BraninFunction
from optimize_options.branin_function_options import branin_function_options

from functions.mccormick_function import McCormickFunction
from optimize_options.mccormick_function_options import mccormick_function_options


def main():
    result = process(McCormickFunction(), mccormick_function_options)
    with open("ans", "w") as file:
        print(result, file=file)
    with open("short_ans", "w") as file:
        for x in result.values():
            x["result"].pop("history")
            print(
                f'X value: {x["result"]["x_min"]}, ' +
                f'distance to min: {x["result"]["min_delta_result"]}'
            )
        print(result, file=file)

if __name__ == "__main__":
    main()

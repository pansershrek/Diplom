from .optimizer import optimize
from .approximate import approximate


def process_optimize(f, optimize_options):
    result = {}
    for idx, options in enumerate(optimize_options):
        result[idx] = {
            "result": optimize(
                f, options["x"], options["opt"], options["eps"],
                f.get_minimum(),
                options.get("max_steps", 10**10)
            )
        }
    return result


def proccess_approximate(f, f_target, approximate_options):
    result = {}
    for idx, options in enumerate(approximate_options):
        result[idx] = {
            "result": approximate(
                f, f_target, options["x"], options["x_validate"],
                options["params"], options["loss_function"],
                options["opt"], options["eps"],
                options.get("max_steps", 10**10)
            )
        }
    return result

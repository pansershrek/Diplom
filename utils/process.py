from .optimizer import optimize
from .approximate import approximate


def process_optimize(f, optimize_options):
    """Minimize target fuction with all options

    :param f: optimize function
    :type f: heir from BaseFunction
    :param optimize_option: all optimize options
    :type optimize_option: list
    :return: result of optimization
    :rtype: dict
    """
    result = {}
    for idx, options in enumerate(optimize_options):
        result[idx] = {
            "result": optimize(
                f, options
            )
        }
    return result


def proccess_approximate(f, f_target, approximate_options):
    """Approximate target fuction with all options

    :param f: approximate function
    :type f: heir from BaseApproximateFunction
    :param f_target: target function
    :type f_target: heir from BaseTargetFunction
    :param approximate_option: all approximate options
    :type approximate_option: dict
    :return: result of approximation
    :rtype: dict
    """
    result = {}
    for idx, options in enumerate(approximate_options):
        result[idx] = {
            "result": approximate(
                f, f_target, options
            )
        }
    return result

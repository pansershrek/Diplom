from .approximate import approximate


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

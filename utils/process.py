from .optimizer import optimize


def process(f, optimize_options):
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

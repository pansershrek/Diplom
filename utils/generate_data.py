import numpy as np

from utils.utils import get_numpy_array


def generate_x(min_border, max_border, num_semples, seed=42, k=4):
    np.random.seed(seed)
    if min_border.shape != max_border.shape:
        raise Exception("Shapes of min and max border is not equal")
    x = (
        (
            max_border - min_border
        ) * np.random.random_sample((num_semples, min_border.shape[0])) + min_border
    )
    return np.around(x, k)


def generate_y(min_border, max_border, num_semples, seed=42, k=4):
    np.random.seed(seed)
    y = (
        (
            max_border - min_border
        ) * np.random.random_sample(num_semples) + min_border
    )
    return np.around(y, k)


def generate_set(x_min_border, x_max_border, y_min_border, y_max_border, num_semples, seed=42):
    return [
        [list(x) for x in generate_x(
            x_min_border, x_max_border, num_semples, seed
        )],
        list(generate_y(
            y_min_border, y_max_border, num_semples, seed
        ))
    ]


def generate_gaussian_noise(num_semples, mean, std, seed=42):
    np.random.seed(seed)
    return np.random.normal(mean, std, num_semples)


def generate_white_noise(num_semples, seed=42):
    np.random.seed(seed)
    return np.random.standard_normal(num_semples)


def generate_salt_and_papper_noise(y, probability_threshold, seed=42):
    np.random.seed(seed)
    y_min = np.min(y)
    y_max = np.max(y)
    noise = np.random.choice(
        [-np.inf, 0, np.inf], p=[
            probability_threshold / 2,
            1 - probability_threshold,
            probability_threshold / 2
        ], size=y.shape
    )
    shifts = []
    for idx in range(y.shape[0]):
        if noise[idx] == 0:
            shifts.append(0)
        elif noise[idx] == np.inf:
            shifts.append(y_max - y[idx])
        else:
            shifts.append(-(y[idx] - y_min))
    return np.array(shifts)


def add_noise(x, y, noise_type, snr=None, mean=0, std=1, probability_threshold=0.5, seed=42):
    if noise_type == "gaussian_noise":
        noise = generate_gaussian_noise(y.shape[0], mean, std, seed)
    elif noise_type == "white_noise":
        noise = generate_white_noise(y.shape[0], seed)
    elif noise_type == "salt_and_papper_noise":
        noise = generate_salt_and_papper_noise(y, probability_threshold, seed)
    elif not(noise_type):
        noise = np.zeros(y.shape[0])
    else:
        raise Exception(f"Invalid noise type: {noise_type}")
    x2y = {}
    if snr is not None:
        y_power = np.mean(y * y)
        noise_power = np.mean(noise * noise)
        new_noise = noise_power * noise / snr
        new_y = y + new_noise
        for xx, yy in zip(x, new_y):
            x2y[str(get_numpy_array(xx))] = yy
    else:
        new_y = y + noise
        for xx, yy in zip(x, new_y):
            x2y[str(get_numpy_array(xx))] = yy
    return x2y

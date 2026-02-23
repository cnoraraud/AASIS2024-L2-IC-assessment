import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import mode
import analysis as ana


def ma(data, properties=None):
    n = properties["n"]
    bg = np.mean(data)
    f = np.ones(n)
    return (
        np.convolve(
            np.pad(
                data,
                (n // 2, n - n // 2 - 1),
                mode="constant",
                constant_values=(bg, bg),
            ),
            f,
            "valid",
        )
        / n
    )


def edge(data, properties=None):
    n = 3
    f = np.array([-1, 0, 1])
    return np.convolve(np.pad(data, (n // 2, n - n // 2 - 1), mode="edge"), f, "valid")


def to_01(data, properties=None):
    axis = -1
    if properties is not None and "axis" in properties:
        axis = properties["axis"]
    pos_data = data - np.nanmin(data, axis=axis, keepdims=True)
    data_s = np.nanmax(pos_data, axis=axis, keepdims=True)
    data_s[np.isclose(data_s, 0.0)] = 1.0
    return pos_data / data_s


def to_0_mode(data, properties=None):
    axis = -1
    if properties is not None and "axis" in properties:
        axis = properties["axis"]
    return data - mode(data, axis=axis, keepdims=True)[0]


def norm(data, properties=None):
    axis = -1
    if properties is not None and "axis" in properties:
        axis = properties["axis"]
    data_c = data - np.nanmean(data, axis=axis, keepdims=True)
    data_s = np.nanmax(np.abs(data_c), axis=axis, keepdims=True)
    data_s[np.isclose(data_s, 0.0)] = 1.0
    return data_c / data_s


def zero_mean(data, properties=None):
    axis = -1
    if properties is not None and "axis" in properties:
        axis = properties["axis"]
    return data - np.nanmean(data, axis=axis, keepdims=True)


def flatten(data, properties=None):
    threshold = np.finfo(np.float64).eps
    if properties is not None and "threshold" in properties:
        threshold = properties["threshold"]
    do_std = False
    if properties is not None and "do_std" in properties:
        do_std = properties["do_std"]
    do_centre = False
    if properties is not None and "do_centre" in properties:
        do_centre = properties["do_centre"]
    reverse = False
    if properties is not None and "reverse" in properties:
        reverse = properties["reverse"]
    side = "pos"
    if properties is not None and "side" in properties:
        side = properties["side"]
    do_pos = side == "pos" or side == "sym" or side == "asym"
    do_neg = side == "neg" or side == "sym" or side == "asym"
    neg_val = 0
    if side == "sym":
        neg_val = 1
    if side == "asym":
        neg_val = -1
    if do_std:
        threshold = threshold * np.nanstd(data)
    centring_value = 0
    if do_centre:
        centring_value = np.median(data)
    flattened_data = np.zeros_like(data)
    if do_pos and reverse:
        flattened_data[
            (data - centring_value >= 0) & (data - centring_value <= threshold)
        ] = 1
    if do_neg and reverse:
        flattened_data[
            (data - centring_value < 0) & (data - centring_value >= -threshold)
        ] = neg_val
    if do_pos and not reverse:
        flattened_data[data - centring_value >= threshold] = 1
    if do_neg and not reverse:
        flattened_data[data - centring_value <= -threshold] = neg_val

    return flattened_data


def to_density(data, properties=None):
    data2d = flatten(np.atleast_2d(data))
    t_max = data2d.shape[1]
    for i in range(data2d.shape[0]):
        starts, ends = ana.get_segments(data2d[i, :])
        for start, end in zip(starts, ends):
            start_bounded = max(start, 0)
            end_bounded = min(end + 1, t_max - 1)
            width = max(0, end_bounded - start_bounded)
            data2d[i, start_bounded:end_bounded] /= width
    density_data = data2d
    if density_data.shape[0] == 1:
        density_data = np.squeeze(density_data, axis=0)
    return density_data


def interpolate_nans(data, properties=None):
    x = np.arange(data.shape[0])
    nans = np.isnan(data)
    f = interp1d(
        x[~nans],
        data[~nans],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    new_data = np.array(data)
    new_data[nans] = f(x[nans])
    return new_data


def fit_to_size(data, properties=None):
    t = properties["t"]
    kind = "nearest"
    if "kind" in properties:
        kind = properties["kind"]
    x = np.arange(data.shape[0]) / data.shape[0]
    x_output = np.arange(t) / t
    f = interp1d(x, data, kind=kind, fill_value="extrapolate", bounds_error=False)
    return f(x_output)

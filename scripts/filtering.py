import numpy as np
from scipy.interpolate import interp1d

def ma(data, properties):
    n = properties["n"]
    bg = np.mean(data)
    f = np.ones(n)
    return np.convolve(np.pad(data, (n//2, n - n//2 - 1), mode='constant', constant_values=(bg, bg)), f, 'valid') / n

def edge(data, properties):
    n = 3
    f = np.array([-1, 0, 1])
    return np.convolve(np.pad(data, (n//2, n - n//2 - 1), mode='edge'), f, 'valid')

def to_01(data, properties):
    pos_data = data - np.min(data)
    return pos_data / np.max(pos_data)

def interpolate_nans(data, properties):
    x = np.arange(data.shape[0])
    nans = np.isnan(data)
    f = interp1d(x[~nans], data[~nans], kind="linear", fill_value='extrapolate', bounds_error=False)
    new_data = np.array(data)
    new_data[nans] = f(x[nans])
    return new_data

def interpolate_to_size(data, properties):
    t = properties["t"]
    x = np.arange(data.shape[0]) / data.shape[0]
    x_output = np.arange(t)/t
    f = interp1d(x, data, kind='nearest', fill_value='extrapolate', bounds_error=False)
    return f(x_output)
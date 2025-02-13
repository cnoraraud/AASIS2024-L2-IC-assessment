import numpy as np

def ma(data, properties):
    n = properties["n"]
    bg = np.mean(data)
    f = np.ones(n)
    return np.convolve(np.pad(data, (n//2, n - n//2 - 1), mode='constant', constant_values=(bg, bg)), f, 'valid') / n

def edge(data, properties):
    n = 3
    f = np.array([-1, 0, 1])
    return np.convolve(np.pad(data, (n//2, n - n//2 - 1), mode='edge'), f, 'valid')
import numpy as np

SPEAKER1 = "S1"
SPEAKER2 = "S2"
SPEAKERS = "[all]"
SPEAKERNONE = "[none]"

def is_string(value):
    return isinstance(value, str) or isinstance(value, np.str_)

def is_speaker_related(speaker, value):
    check_substring = value.split(" ")[0]
    if speaker in check_substring: return True
    if SPEAKERS in check_substring: return True
    return False

def get_speakers():
    return [SPEAKER1, SPEAKER2, SPEAKERS]

def get_speaker_other(speaker):
    if speaker == SPEAKER1:
        return SPEAKER2
    if speaker == SPEAKER2:
        return SPEAKER1
    return SPEAKERNONE

def count_lengths(list_of_a):
    counts = []
    for a in list_of_a:
        counts.append(count(a))
    return np.array(counts)

def sum_lengths(list_of_a):
    sums = []
    for a in list_of_a:
        sums.append(sum_data(a))
    return np.array(sums)

def valid(data):
    if data is None: return False
    if np.isscalar(data):
        if np.isnan(data): return False
        if np.isinf(data): return False
    if isinstance(data, np.ndarray) and data.shape[0] == 0: return False
    if isinstance(data, list) and len(data) == 0: return False
    return True

def quantiles(data, q, method="linear"):
    if not valid(data):
        return None
    return np.nanquantile(data, q=q, method=method)

def group_arrays(arrays):
    if len(arrays) == 0:
        return np.empty((0))
    return np.concat(arrays, axis=0)

def count(data):
    if not valid(data):
        return 0
    return data.size - np.isnan(data).sum()

def sum_data(data):
    if not valid(data):
        return 0
    return np.sum(data)

def mean_data(data):
    if not valid(data):
        return np.nan
    return np.nanmean(data)

def median_data(data):
    if not valid(data):
        return np.nan
    return np.nanmedian(data)

def std_data(data):
    if not valid(data):
        return np.nan
    return np.nanstd(data)

def var_data(data):
    if not valid(data):
        return np.nan
    return np.nanvar(data)

def div_datas(data1, data2):
    if not valid(data1) or not valid(data2):
        return np.nan
    return data1/data2

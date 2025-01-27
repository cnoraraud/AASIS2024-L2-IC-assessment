import numpy as np
import io_tools as iot
import math
import sys

def nothas(labels, sub):
    return np.char.find(np.array(labels), sub) == -1

def has(labels, sub):
    return np.char.find(np.array(labels), sub) != -1

def select_labels(data, labels, label_filter):
    return data[label_filter, :], labels[label_filter]

def reorder_data(data, labels):
    order = np.argsort(labels, stable='True')
    return data[order, :], labels[order]

def select_time(data, time_filter):
    return data[:, time_filter]

def append_labels(data, labels, new_data, new_labels):
    return np.concat([data, new_data]), np.concat([labels, new_labels])

def pmean(d, p=1):
    # found out scipy has this after I wrote it
    n = d.shape[0]
    dm = d
    if p == 0:
        dm = np.prod(d, axis=0, keepdims=True) ** (1 / n)
    elif math.isinf(p) and p > 0:
        dm = np.max(d, axis=0, keepdims=True)
    elif math.isinf(p) and p < 0:
        dm = np.min(d, axis=0, keepdims=True)
    else:
        dm = (np.sum(d**p, axis=0, keepdims=True)/n) ** (1/p)
    return dm, p

def write_DL(name, D, L):
    npzs_path = iot.npzs_path()
    path = npzs_path / name
    np.savez(path, D=D, L=L)
    return path

def npz_list():
    npzs = []
    for name in iot.list_dir(iot.npzs_path(), "npz"):
        npzs.append(name)
    return npzs

def print_npz():
    for name in npz_list():
        print(name)

def read_DL_from_name(name):
    if ".npz" not in name:
        name = name + ".npz"
    return read_DL(iot.npzs_path() / name)

def read_DL(path):
    npz = np.load(path)
    D = npz['D']
    L = npz['L']
    name = path.stem
    return name, D, L

def data_head(table, n = None):
    return data_section(table, n)

def data_section(table, start = 0, n = None):
    if n is None:
        n = len(table) - start
    return dict(list(table.items())[start : start + n]) # dictionaries are ordered as of python 3.7

def read_DLs(table, key_whitelist=None):
    all_names = []
    all_Ds = []
    all_Ls = []
    for key in table:
        if key_whitelist is None or key in key_whitelist:
            row = table[key]
            name, D, L = read_DL_from_name(key)
            all_names.append(name)
            all_Ds.append(D)
            all_Ls.append(L)
    return all_names, all_Ds, all_Ls

def get_DLs(table = None, n = None, start:int = 0):
    if table is None:
        table = generate_npz_table()
    return read_DLs(table = data_section(table, start, n))

def generate_npz_table():
    npzs = npz_list()
    table = dict()
    for npz in npzs:
        table[npz] = dict({"name": None, "D": None, "L": None})
    return table

class NpzProvider:
    npzs = []
    table = dict()

    def __len__(self):
        return len(self.npzs)
    
    def __init__(self):
        self.npzs = npz_list()
        self.table = generate_npz_table()
    
    def reset(self):
        self.pointer = 0

    def generator(self, n = 1):
        i = 0
        while i < len(self):
            yield get_DLs(self.table, n = n, start = i)
            i += n
    
    def nth_key(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        return list(rows.keys())[n]
    
    def nth(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        return rows[self.nth_key(rows, n)]
    
    def report_rows(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        for key in rows:
            row = rows[key]
            print(f"[{key}]")
        
if __name__ == '__main__':
    globals()[sys.argv[1]]()
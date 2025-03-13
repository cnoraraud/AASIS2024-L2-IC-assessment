import sys
import pathlib as p
import numpy as np
import numpy_wrapper as npw
import io_tools as iot
import pandas as pd
from scipy import interpolate

# TODO: similar function in filtering.py... combine somehow?
def fit_to_data(x,y,t_max=None, kind="linear"):
    if t_max is None:
        t_max = x.max()
    f = interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")
    new_x = np.arange(t_max)
    new_y = f(new_x)
    return new_x, new_y

def time_string_to_ms(time_string):
    slots = time_string.split(":")
    mults = [60000, 1000, 1]
    ms = 0.0
    for i in range(len(slots)):
        mult = mults[i]
        slot = slots[i]
        val = 0
        if slot != "":
            val = float(slot)
        ms += val * mult    
    return ms

def get_joystick_data(js, t_max=None):
    x = []
    y = []
    with open(iot.joystick_csvs_path() / js, "r") as f:
        while line := f.readline():
            line = line.strip()
            if line[0] == "#":
                continue
            values = line.split(",")
            t = round(float(values[0].strip())*1000)
            x_js = int(values[1].strip())
            y_js = int(values[2].strip())
            f_js = int(values[3].replace(";","").strip())
            features = [x_js, y_js, f_js]
            x.append(t)
            y.append(features)
    x = np.array(x)
    y = np.array(y)
    fires = np.where(y[:,2]==1)[0]
    valid_range = range(fires[0], fires[1])
    x = x[valid_range]
    y = y[valid_range,0:2]
    x = x - x.min()
    y = (y-500)/500
    xs = []
    ys = []
    for i in range(y.shape[1]):
        x_new, y_new = fit_to_data(x,y[:,i],t_max=t_max)
        xs.append(x_new)
        ys.append(y_new)
    y = np.concatenate([ys])
    return y

def get_pyfeat_data(csv_name, t_max=None):
    # LOAD CSV
    df = pd.read_csv(iot.pyfeat_csvs_path() / csv_name)

    # REMOVE EMPTY COLUMNS
    bad_cols = []
    for col in df.columns:
        if (df[col].dtype == "object"):
            #print(f"Object column found \'{col}\'")
            continue
        if np.sum(~np.isnan(df[col])) == 0:
            bad_cols.append(col)
    df = df.drop(columns=bad_cols)

    # CALCULATE MS
    approx_ms = []
    for row in df[["approx_time"]].iterrows():
        i = row[0]
        row_data = row[1]
        approx_time = row_data["approx_time"]
        approx_ms.append(time_string_to_ms(approx_time))
    df["approx_ms"] = approx_ms
    fpms = (df["frame"]/(df["approx_ms"] + sys.float_info.epsilon)).median()
    fps = int(fpms * 1000)
    mspf = 1/fpms
    df["ms"] = np.round(df["approx_ms"] + (df["frame"] % fps + 0.5) * mspf).astype(int)

    # FIND AVERAGE X&Y AND AUS
    x_cols = []
    y_cols = []
    AU_cols = []
    for col in df.columns:
        name = col.split("_")[0]
        if name == "x":
            x_cols.append(col)
        if name == "y":
            y_cols.append(col)
        if "AU" in name:
            AU_cols.append(col)
    df_X = df[x_cols]
    df_Y = df[y_cols]
    mean_x = df_X.mean(axis=1)
    mean_y = df_Y.mean(axis=1)
    df["mean_x"] = mean_x
    df["mean_y"] = mean_y

    # FINDING RELEVANT COLUMNS
    emotion_cols = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    position_cols = ["mean_x", "mean_y", "Pitch", "Roll", "Yaw"]
    good_cols = position_cols + emotion_cols + AU_cols
    df_ms = df["ms"]
    df_good = df[good_cols]

    # CONVERT TO DL framework
    t = df_ms.to_numpy()
    D_raw = df_good.to_numpy().T
    L = npw.string_array(df_good.columns.to_list())

    # CONVERT TO SHAPE
    if isinstance(t_max, type(None)):
        t_max = t.max()
    new_shape = (L.shape[0], t_max)
    D = np.full(new_shape, 0.0)
    for i in range(L.shape[0]):
        d = D_raw[i,:]
        not_nan_mask = ~np.isnan(d)
        _, new_d = fit_to_data(t[not_nan_mask], d[not_nan_mask], t_max=t_max)
        D[i,:] = new_d
    return D, L
            

def add_facial_features_to_data(name, D, L):
    Ds = [D]
    Ls = [L]
    for csv_path in iot.get_csv_names_facial_features():
        csv_name = p.Path(csv_path).name
        ff_D, ff_L = get_pyfeat_data(csv_name, t_max=D.shape[1])
        Ds.append(ff_D)
        Ls.append(ff_L)

    D_concat = np.concat(Ds, axis=0)
    L_concat = np.concat(Ls, axis=0)
    return name, D_concat, L_concat

# TODO: Refactor get_joysticks_for_D with fuzzy features like in get_facial_features_for_D

def add_joysticks_to_data(name, D, L):
    new_D = np.copy(D)
    new_L = np.copy(L)
    stem_name = "_".join(name.split("_")[:-1])
    t_max = D.shape[1]
    series = []
    labels = []
    for csv_name in iot.get_csv_names_joystick(stem_name):
        if csv_name not in iot.list_dir(iot.csvs_path(), "csv"):
            continue
        csv_values = csv_name.split("_")
        speaker = csv_values[-3][-3:]
        annotator = csv_values[-2]
        version = csv_values[-1][:3]
        new_Lx = f"{speaker} (ann.) js:x {annotator} {version}"
        new_Ly = f"{speaker} (ann.) js:y {annotator} {version}"
        joystick_data = get_joystick_data(csv_name, t_max)
        x_data = joystick_data[0,:]
        y_data = joystick_data[1,:]
        if new_Lx not in L:
            series.append(x_data)
            labels.append(new_Lx)
        else:
            new_D[np.where(L == new_Lx)[0][0],:] = x_data
        if new_Ly not in L:
            series.append(y_data)
            labels.append(new_Ly)
        else:
            new_D[np.where(L == new_Ly)[0][0],:] = y_data
    if len(series) > 0 and len(labels) > 0:
        new_D = np.concat([new_D, np.array(series)], axis=0)
        new_L = np.concat([L, np.array(labels)], axis=0)
    return name, new_D, new_L
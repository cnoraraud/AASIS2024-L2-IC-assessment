import sys

import data_logger as dl
import io_tools as iot
import naming_tools as nt
import numpy as np
import numpy_wrapper as npw
import pandas as pd
from scipy.interpolate import interp1d


def pyfeat_names():
    names = []
    for name in iot.list_dir_names(iot.pyfeat_csvs_path(), "csv"):
        names.append(name)
    return names


def joystick_names():
    names = []
    for name in iot.list_dir_names(iot.joystick_csvs_path(), "csv"):
        names.append(name)
    return names


def fit_to_data(x, y, t_max=None, kind="linear"):
    if t_max is None:
        t_max = x.max()
    f = interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")
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


def pd_to_DL(df_ms, df_data, t_max=None):
    # CONVERT TO DL framework
    t = df_ms.to_numpy()
    D_raw = df_data.to_numpy().T
    L = npw.string_array(df_data.columns.to_list())

    # CONVERT TO SHAPE
    if isinstance(t_max, type(None)):
        t_max = t.max()
    new_shape = (L.shape[0], t_max)
    D = np.full(new_shape, 0.0)
    for i in range(L.shape[0]):
        d = D_raw[i, :]
        not_nan_mask = ~np.isnan(d)
        _, new_d = fit_to_data(t[not_nan_mask], d[not_nan_mask], t_max=t_max)
        D[i, :] = new_d
    return D, L


def get_joystick_data(csv_name, t_max=None):
    df = pd.read_csv(
        iot.joystick_csvs_path() / csv_name,
        skip_blank_lines=True,
        comment="#",
        header=None,
    )
    df.columns = ["time", "x", "y", "f"]
    df["ms"] = np.round(df["time"] * 1000).astype(int)
    valid_range = df[df["f"] == 1]["ms"].tolist()
    if len(valid_range) >= 2:
        df = df[(df["ms"] > valid_range[0]) & (df["ms"] < valid_range[-1])]

    good_cols = ["x", "y"]
    df_ms = df["ms"]
    df_good = df[good_cols]

    return pd_to_DL(df_ms, df_good, t_max)


def get_pyfeat_data(csv_name, t_max=None):
    # LOAD CSV
    df = pd.read_csv(iot.pyfeat_csvs_path() / csv_name)

    # REMOVE EMPTY COLUMNS
    bad_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        if np.sum(~np.isnan(df[col])) == 0:
            bad_cols.append(col)
    df = df.drop(columns=bad_cols)

    # CALCULATE MS
    approx_ms = []
    for _, row_data in df[["approx_time"]].iterrows():
        approx_time = row_data["approx_time"]
        approx_ms.append(time_string_to_ms(approx_time))
    df["approx_ms"] = approx_ms
    fpms = (df["frame"] / (df["approx_ms"] + sys.float_info.epsilon)).median()
    fps = int(fpms * 1000)
    mspf = 1 / fpms
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
    emotion_cols = [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
    ]
    position_cols = ["mean_x", "mean_y", "Pitch", "Roll", "Yaw"]
    good_cols = position_cols + emotion_cols + AU_cols
    df_ms = df["ms"]
    df_good = df[good_cols]

    return pd_to_DL(df_ms, df_good, t_max)


def add_facial_features_to_data(data_name, D, L):
    Ds = [D]
    Ls = [L]
    t_max = D.shape[1]
    for csv_path in iot.get_csv_names_facial_features(data_name):
        ff_D, ff_L = get_pyfeat_data(csv_path, t_max=t_max)

        name = nt.file_swap(nt.get_name(csv_path))
        canidates = nt.find_best_candidate(nt.find_speakers(name))
        if len(canidates) > 1:
            dl.log(f"Mysterious file name ({name}), multiple speakers share a face?")
            continue

        source = nt.compact_sources(nt.speakers_to_sources(canidates))
        tag = nt.ANNOTATION_TAG

        new_l = []
        for feature in ff_L:
            feature_clean = feature.lower()
            new_l.append(nt.create_label(source, tag, f"{nt.FACIAL_FEATURE_TYPE}:{feature_clean}"))
        Ds.append(ff_D)
        Ls.append(npw.string_array(new_l))

    D_concat = np.concat(Ds, axis=0)
    L_concat = np.concat(Ls, axis=0)
    return data_name, D_concat, L_concat


def add_joysticks_to_data(data_name, D, L):
    Ds = [D]
    Ls = [L]
    t_max = D.shape[1]
    for csv_path in iot.get_csv_paths_joystick(data_name):
        js_D, js_L = get_joystick_data(csv_path, t_max=t_max)

        name = nt.file_swap(nt.get_name(csv_path))
        canidates = nt.find_best_candidate(nt.find_speakers(name))
        if len(canidates) > 1:
            dl.log(
                f"Mysterious file name ({name}), multiple speakers share a joystick annotation?"
            )
            continue

        source = nt.compact_sources(nt.speakers_to_sources(canidates))
        tag = nt.ANNOTATION_TAG
        annotator = nt.find_annotator(name)
        version = nt.find_version(name)
        extra = f"{version}"

        new_l = []
        for feature in js_L:
            feature_clean = feature.lower()
            new_l.append(
                nt.create_label(source, tag, f"{nt.JOYSTICK_TYPE}:{feature_clean}-{annotator}", extra)
            )
        Ds.append(js_D)
        Ls.append(npw.string_array(new_l))

    D_concat = np.concat(Ds, axis=0)
    L_concat = np.concat(Ls, axis=0)
    return data_name, D_concat, L_concat

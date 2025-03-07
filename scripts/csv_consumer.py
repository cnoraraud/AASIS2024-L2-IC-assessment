import numpy as np
import io_tools as iot
from scipy import interpolate

# TODO: similar function in filtering.py... combine somehow?
def fitToData(x,y,t_max=None):
    if t_max is None:
        t_max = x.max()
    f = interpolate.interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    new_x = np.arange(t_max)
    new_y = f(new_x)
    return new_x, new_y

def joystickData(js, t_max=None):
    x = []
    y = []
    with open(iot.csvs_path() / js, "r") as f:
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
        x_new, y_new = fitToData(x,y[:,i],t_max=t_max)
        xs.append(x_new)
        ys.append(y_new)
    y = np.concatenate([ys])
    return y

def get_joysticks_for_D(name, D, L):
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
        joystick_data = joystickData(csv_name, t_max)
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
import torch
from torch.nn import functional as F
import io_tools as iot
from datetime import datetime
import data_logger as dl
import npz_reader as npzr
import pandas as pd
import os
import numpy as np
import numpy_wrapper as npw
import dataframe_sourcer as dfs
import analysis as ana
import filtering as filt
import keywords_recipes as kwr

def get_dataset_distribution(dataset):
    ys = dataset.chart_labels["y"]
    yo = dataset.chart_labels["yo"]
    y = pd.concat([ys, yo], ignore_index=True)
    y = y.fillna(y.mean())
    return get_y_distribution(y)

def get_y_distribution(y):
    counts = y.round().astype("int").value_counts()
    all_counts = [0, 0, 0, 0, 0, 0]
    for i, grade in enumerate([1, 2, 3, 4, 5, 6]):
        if grade in counts:
            all_counts[i] = counts[grade].item()
    return all_counts

def get_weights_from_distribution(distribution, type="p", k=1):
    w = np.array(distribution)
    w_sum = np.nansum(w)
    if type == "p":
        w = 1.0 - w/w_sum
    elif type == "cum_p":
        w = np.nancumsum(w)[:-1]
        w = w/w_sum
    elif type == "bal":
        w = np.nan_to_num(np.divide(w_sum - w, np.clip(len(w) * w, 1e-6, None)))

    if type == "p" or type == "cum_p":
        w_mean = np.nanmean(w)
        w = w/w_mean
        weights = np.power(w, k)
    elif type == "bal":
        weights = w
    return weights

def preprocess_for_nn(D, reduce_factor=5, filter_width=2500, do_norm=False, do_ma=False):
    long_factor = 0.3
    peak_factor = 0.3

    new_length = D.shape[1] // reduce_factor
    if do_norm:
        D = filt.to_0_mode(filt.norm(D))
    if do_ma:
        if do_norm:
            D_peak = np.power(np.copy(D), 3)
        D_long_ma = ana.apply_method_to_all_features(D, filt.ma, {"n":filter_width * 4})
        for i in range(3):
            D = ana.apply_method_to_all_features(D, filt.ma, {"n":filter_width})
        
        D = D * (1.0 - long_factor) + D_long_ma * long_factor
        if do_norm:
            D = D * (1.0 - peak_factor) + D_peak * peak_factor
    if reduce_factor != 1:
        D = ana.apply_method_to_all_features(D, filt.fit_to_size, {"t":new_length, "destructive":True, "kind": "nearest"})
        if do_norm:
            D = filt.norm(D)
    return D 

def create_chart_segments(D, S, SO, y, yo, task, annotator, standard_chunk=2500, reduce_factor=1, source=None, side=None, do_norm=False, do_ma=False):
    T = D.shape[1]
    N = (T + standard_chunk) // standard_chunk

    rows = []
    for i in range(N):
        start_i = max(0, i * standard_chunk)
        end_i = min(T, (i + 1) * standard_chunk)
        data_slice = torch.tensor(D[:, start_i:end_i])
        channels = data_slice.shape[0]
        length = data_slice.shape[1]
        file_name = f"{task}_{annotator}_{reduce_factor}_{standard_chunk}_{S}_{i}.pt"
        torch.save(data_slice, iot.chart_pts_path() / file_name)
        row = {"file_name": file_name,
               "item_id": f"{S}{side}_{task}_{annotator}",
               "task": task,
               "annotator": annotator,
               "side": side,
               "speaker": S,
               "interlocutor": SO,
               "channels": channels,
               "length": length,
               "max_length": standard_chunk,
               "reduce_factor": reduce_factor,
               "i": i,
               "start_i": start_i,
               "end_i": end_i,
               "last_i": T-1,
               "y": y,
               "yo": yo,
               "source": source,
               "time": datetime.now()
               }
        rows.append(row)
        dl.write_to_manifest_new_file(dl.CHART_PREPROCESSING_TYPE, iot.DL_npzs_path() / source, iot.chart_pts_path() / file_name, output_info=data_slice.size())
    extra = ""
    if do_norm:
        extra = f"{extra}_norm"
    if do_ma:
        extra = f"{extra}_ma"
    chart_file_name = f"chart_labels_{reduce_factor}_{standard_chunk}{extra}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(iot.label_csvs_path() / chart_file_name, mode="a", sep="\t", index=False, header=not os.path.isfile(iot.label_csvs_path() / chart_file_name))
    return df

def get_speaker_Ds(D, L, L_c):
    D_c = np.full((L_c.shape[0], D.shape[1]), np.nan)
    D_c[np.isin(L_c, L)] = D

    DL1, DL2, DLA = npzr.extract_speakers_DLs(D_c, L_c)
    expected_shape = D_c.shape
    del D_c
    D1, _ = DL1
    D2, _ = DL2
    DA, _ = DLA
    D_A = np.concatenate([D1,D2,DA])
    D_B = np.concatenate([D2,D1,DA])
    del D1
    del D2
    del DA
    
    #sanity checks
    if (expected_shape != D_A.shape) or (expected_shape != D_B.shape):
        raise Exception("Shapes don't match")
    
    return D_A, D_B

def extract_chart_segments(session, L_c, reduce_factor = 1, standard_chunk = 2500, do_norm=False, do_ma=False):
    npz = session["npz"]
    y_1 = session["holistic_1"]
    y_2 = session["holistic_2"]
    S1 = session["speaker_1"]
    S2 = session["speaker_2"]
    annotator = session["annotator"]
    task = session["task"]

    name, D, L = npzr.read_sanitized_DL_from_name(npz)
    D, L = npzr.reorder_data(D, L)

    # Preprocessing
    D = preprocess_for_nn(D, reduce_factor=reduce_factor, filter_width=standard_chunk, do_norm=do_norm, do_ma=do_ma)
    if reduce_factor != 1:
        standard_chunk = standard_chunk // reduce_factor

    # Storing
    D_A, D_B = get_speaker_Ds(D, L, L_c)
    df_A = create_chart_segments(D_A, S1, S2, y_1, y_2, task, annotator, standard_chunk=standard_chunk, reduce_factor=reduce_factor, source=npz, side="A")
    df_B = create_chart_segments(D_B, S2, S1, y_2, y_1, task, annotator, standard_chunk=standard_chunk, reduce_factor=reduce_factor, source=npz, side="B")
    
    del D_A
    del D_B
    dl.write_to_manifest_log(dl.CHART_PREPROCESSING_TYPE, f"Finished chart extraction for {npz}")

def create_all_chart_datas(tasks, reduce_factor=1, standard_chunk = 2500, do_norm=False, do_ma=False):
    channels, total_channels = npzr.get_all_channels()
    L_c = npw.string_array(total_channels)
    
    speakers, samples = dfs.get_speakers_samples_full_dataframe(tasks)
    session_df = samples[["npz","holistic_1","holistic_2","speaker_1","speaker_2", "task", "annotator"]]
    
    dl.write_to_manifest_log(dl.CHART_PREPROCESSING_TYPE, f"Extracting charts", level=0)
    for i in range(len(session_df)):
        session = session_df.iloc[i]
        extract_chart_segments(session, L_c, reduce_factor=reduce_factor, standard_chunk=standard_chunk, do_norm=do_norm, do_ma=do_ma)

class ChartDataset(torch.utils.data.Dataset):
    def __init__(self, recipe, allowed_speakers=None, n=None, e=None, transform=None, target_transform=None):
        self.labels_dir = recipe.get(kwr.labels_dir) or iot.label_csvs_path()
        self.charts_dir = recipe.get(kwr.charts_dir) or iot.chart_pts_path()

        df_labels = pd.read_csv(self.labels_dir / recipe.get(kwr.label_file), sep="\t")
        if not isinstance(allowed_speakers, type(None)):
            speaker_cond = df_labels["speaker"].isin(allowed_speakers)
            interlocutor_cond = df_labels["interlocutor"].isin(allowed_speakers)
            cond = speaker_cond & interlocutor_cond
            df_labels = df_labels[cond]
        df_labels["time"] = df_labels["time"].astype("datetime64[ns]")
        df_labels = df_labels.sort_values("time").drop_duplicates(["item_id", "i"], keep="last").sort_values(["item_id", "i"]).reset_index().drop(columns=["index"]).reset_index()
        self.chart_labels = df_labels
        
        self.transform = transform
        self.target_transform = target_transform
        self.context = recipe.get(kwr.context) or 1
        self.hop = recipe.get(kwr.hop) or 1
        self.index_map = self.get_index_map()
        self.unit = self.chart_labels["length"].max().item()
        self.channels = self.chart_labels["channels"].max().item()
        self.n = n
        self.e = e
        self.do_pad = (not isinstance(self.n, type(None))) or (not isinstance(self.e, type(None)))
        self.nan = 0
        self.dtype = recipe.get(kwr.dtype) or torch.float64
        self.device = recipe.get(kwr.device)
        self.add_time = recipe.get(kwr.add_time) or False
        self.transpose_x = recipe.get(kwr.transpose_x) or False
        self.y_n = recipe.get(kwr.y_n) or -1
    
    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, idx):
        charts = []
        start_i, end_i = self.index_map[idx]
        for i in range(start_i, end_i + 1, 1):
            file_name = self.chart_labels.iloc[i, 1]
            chart = torch.load(self.charts_dir / file_name)
            charts.append(chart)
        chart = torch.cat(charts, dim=1)
        length = chart.shape[1]
        if self.add_time:
            chart_start = self.chart_labels.iloc[start_i, 14]
            chart_max = self.chart_labels.iloc[start_i, 15]
            time_channel = (torch.arange(length).unsqueeze(0) + chart_start) /  chart_max
            chart = torch.cat((chart, time_channel), dim=0)
        if self.do_pad:
            n_pad = 0
            e_pad = 0
            if not isinstance(self.n, type(None)):
                n_pad = self.n - chart.shape[1]
                chart = F.pad(chart, (0, n_pad), value=self.nan)
            if not isinstance(self.e, type(None)):
                e_pad = self.e - chart.shape[0]
                chart = F.pad(chart, (0, 0, 0, e_pad), value=self.nan)
        y0 = self.chart_labels.iloc[start_i, 16]
        y1 = self.chart_labels.iloc[start_i, 17]
        ys = (y0, y1)
        if self.y_n != -1:
            ys = ys[:self.y_n]
        label = torch.tensor(ys)
        chart = chart.to(dtype=self.dtype, device=self.device)
        label = label.to(dtype=self.dtype, device=self.device)
        if not isinstance(self.transform, type(None)):
            chart = self.transform(chart)
        if not isinstance(self.target_transform, type(None)):
            label = self.target_transform(label)
        chart = chart.nan_to_num()
        if self.transpose_x:
            chart = chart.transpose(-1, -2)
        return chart, label, length

    def get_index_map(self):
        item_group = self.chart_labels.groupby("item_id", sort=False, dropna=False)
        start_is = item_group["index"].min().to_list()
        end_is = item_group["index"].max().to_list()
        hop_is = ((item_group["i"].max() + self.hop - self.context + 1) // self.hop).to_list()
        idx_map = dict()
        idx = 0
        for start_i, end_i, hop_n in zip(start_is, end_is, hop_is):
            for j in range(hop_n):
                start_id = start_i + self.hop * j
                end_id = start_i + self.hop * j + self.context - 1
                end_id = min(end_id, end_i)
                idx_map[idx] = [start_id, end_id]
                idx += 1
        return idx_map

    def get_embed_dims(self):
        n = self.context * self.unit
        e = self.channels + int(self.add_time)
        return n, e

def collate_nested(batch):
    inp_transposed, tgt_transposed, len_transposed = list(zip(*batch))
    inp_untransposed = []
    for inp in inp_transposed:
        inp_untransposed.append(inp.T.contiguous())
    inp = torch.nested.nested_tensor(inp_untransposed, layout=torch.jagged).contiguous()
    tgt = torch.stack(tgt_transposed, 0)
    lens = torch.tensor(len_transposed)
    return inp, tgt, lens

def collate_stack(batch):
    inp_transposed, tgt_transposed, len_transposed = list(zip(*batch))
    inp = torch.stack(inp_transposed, 0).transpose(1, 2)
    tgt = torch.stack(tgt_transposed, 0)
    lens = torch.tensor(len_transposed)
    return inp, tgt, lens
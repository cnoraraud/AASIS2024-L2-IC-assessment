import npz_reader as npzr
import naming_tools as nt
import dataframe_sourcer as dfs
import numpy as np
import population_analyzer as popa
import data_displayer as dd
import filtering as filt
from itertools import combinations
import pandas as pd
import io_tools as iot
import data_logger as dl

def get_groups_and_annotators(samples, color_k=2):
    speakers_of_interest = dfs.get_annotations_anchor_speakers()
    anchor_samples = samples[(samples["speaker_1"].isin(speakers_of_interest)) | (samples["speaker_2"].isin(speakers_of_interest))][["npz","speaker_1","speaker_2","task","annotator"]]
    anchor_samples["name"] = anchor_samples["speaker_1"].astype(str) + "-" + anchor_samples["speaker_2"].astype(str) + ":" + anchor_samples["task"].astype(str)
    anchor_samples = anchor_samples.drop(columns=["speaker_1", "speaker_2", "task"]).sort_values("name")
    
    unique_annotators = anchor_samples["annotator"].unique().tolist()
    unique_groups = anchor_samples["name"].unique().tolist()
    unique_colors = colors_simple_tetradic()
    annotator_dict = dict()
    for unique_annotator, rgb in zip(unique_annotators, unique_colors[:len(unique_annotators)]):
        annotator_dict[unique_annotator] = {"name": unique_annotator, "rgb": rgb}
    
    for annotator_combination in combinations(unique_annotators, 2):
        rgbs = [] 
        names = [] 
        for annotator in annotator_combination:
            names.append(annotator_dict[annotator]["name"])
            rgbs.append(annotator_dict[annotator]["rgb"])
        combined_name = " + ".join(sorted(names))
        combined_rgb = intmix(*rgbs, color_k=color_k)
        annotator_dict[combined_name] = {"name": combined_name, "rgb": combined_rgb}
    return anchor_samples, unique_groups, annotator_dict, unique_annotators

def colors_ocean():
    ocean_blue = np.array([38,84,125])
    ocean_red = np.array([239,71,111])
    ocean_yellow = np.array([255,209,102])
    ocean_green = np.array([6,214,160])
    return [ocean_blue, ocean_red, ocean_yellow, ocean_green]

def colors_simple_tetradic():
    green = np.array([0,255,50])
    blue = np.array([0,77,255])
    magenta = np.array([255,0,205])
    orange = np.array([255,178,0])
    return [green, blue, magenta, orange]

# cool in practice, doesn't worth with arrays ;-; but i'll keep it here
def intmix(*args, weights=None, color_k=1):
    if isinstance(weights, type(None)):
        n = len(args)
        args = np.power(args, color_k)
        res = np.sum(args,axis=0)
    else:
        background_color = np.array([0,0,0])
        args = list(args) + [background_color]
        args = np.power(args, color_k)
        background_value = max(min(1.0 - np.sum(weights), 1.0), 0)
        weights = np.expand_dims(list(weights) + [background_value], 0)
        res = (np.array(weights) @ args).flatten()
    res = np.power(res, 1/color_k)
    res = np.rint(res).astype('int')
    res = np.clip(res, 0, 255)
    return res

def get_group_dif_DL(group, annotator_dict):
    Ds = []
    Ls = []
    shapes = []
    N = len(group)
    for i in range(N):
        annotation = group.iloc[i]
        name, D, L  = npzr.read_DL_from_name(annotation["npz"])
        D, L = npzr.reorder_data(D, L)
        Ds.append(D)
        Ls.append(L)
        shapes.append(D.shape)
    
    L_c = popa.find_total_union(Ls)
    C = len(L_c)
    T = np.max(shapes, axis=0)[1]
    layer_shape = [C, T]
    final_shape = [N] + layer_shape
    time_axis = np.full(T, True)
    D_c = np.zeros(final_shape)
    for i in range(N):
        D = Ds[i]
        L = Ls[i]
        indexes = np.ix_(np.isin(L_c, L), time_axis)
        D_new = np.zeros(layer_shape)
        D_new[indexes] = D
        D_c[i] = D_new
        del D
        del D_new
    del Ds
    
    mins = np.nanmin(np.nanmin(D_c, axis=2, keepdims=True), axis=0, keepdims=True)
    D_c = D_c - mins
    maxes = np.nanmax(np.nanmax(D_c, axis=2, keepdims=True), axis=0, keepdims=True)
    not_zero = ~np.isclose(maxes, 0)
    more_than_one = maxes > 1
    div_condition = not_zero & more_than_one
    D_c = np.divide(D_c, maxes, out=D_c, where=div_condition)
    return D_c, L_c

def get_group_dif_rgb_DL(group, annotator_dict, color_k = 1):
    D, L = get_group_dif_DL(group, annotator_dict)
    D = np.expand_dims(D, 3)
    colors = []
    N = len(group)
    for i in range(N):
        annotation = group.iloc[i]
        annotator = annotation["annotator"]
        color = annotator_dict[annotator]["rgb"]
        colors.append(color)
    C = D.shape[1]
    T = D.shape[2]
    layer_shape = [C, T]
    color_shape = layer_shape + [3]
    final_color_shape = [N] + color_shape
    D_rgb = np.full(final_color_shape, 0)
    for i in range(N):
        D_rgb[i] = np.full(color_shape, colors[i])
    
    
    D = filt.flatten(D)
    for i in range(N):
        D_rgb[i] = np.power(D_rgb[i] * D[i], color_k)
    #D_rgb = C_m.mean(axis=0)
    D_rgb = D_rgb.sum(axis=0)
    D_rgb = np.power(D_rgb, 1/color_k)
    D_rgb = np.clip(D_rgb, 0, 255)
    D_rgb = np.rint(D_rgb).astype(int)
    
    del D

    return D_rgb, L

def create_figures(anchor_samples, unique_groups, annotator_dict, color_k=2):
    for i in range(len(unique_groups)):
        group_name = unique_groups[i]
        group = anchor_samples[anchor_samples["name"] == group_name]
        annotator_len = len(group["annotator"].unique())
        D, L = get_group_dif_rgb_DL(group, annotator_dict, color_k = color_k)
        D, L = npzr.do_label_select(D, L, npzr.has(L, "(ann.)") & npzr.nothas(L, "ff:") & npzr.nothas(L, "js:"))
        subfolder = "annotator_comparison"
        dd.cc(D, L, f"Annotator overlap in \"{group_name}\"", norm=None, color_dict=annotator_dict, savefig=True, subfolder=subfolder)
        dl.write_to_manifest_new_summary_file(
            dl.STATISTICS_TYPE,
            iot.figs_path()/subfolder,
            f"figure:{group_name}",
            f"{annotator_len}",
        )
        del D
        del L

def create_table(anchor_samples, unique_groups, annotator_dict, unique_annotators):
    channel_dict = dict()
    for i in range(len(unique_groups)):
        group_name = unique_groups[i]
        group = anchor_samples[anchor_samples["name"] == group_name]
        group_annotators = group["annotator"].tolist()
        indexable = np.isin(unique_annotators, group_annotators)
        D, L = get_group_dif_DL(group, annotator_dict)
        channel_corrs = []
        channel_sums = []
        label_filter = npzr.has(L, "(ann.)") & npzr.nothas(L, "ff:") & npzr.nothas(L, "js:")
        for c in range(D.shape[1]):
            if not label_filter[c]:
                continue
            channel_corr = np.corrcoef(D[:,c,:])
            channel_sum = np.nanmean(D[:,c,:], axis=1)
            np.fill_diagonal(channel_corr, np.nan)
            channel_corrs.append(channel_corr)
            channel_sums.append(channel_sum)
        del D
        
        for corrs, sums, l in zip(channel_corrs, channel_sums, L[label_filter].tolist()):
            label_name = l.split(" ")[-1]
            channel_annotator_corrs = np.full(len(unique_annotators), np.nan)
            channel_annotator_sums = np.full(len(unique_annotators), np.nan)
            channel_annotator_corrs[indexable] = np.nanmean(corrs, axis=0)
            channel_overall_sums = np.nanmean(sums)
            channel_annotator_sums[indexable] = (sums - channel_overall_sums) / channel_overall_sums
            channel_overall_corrs = np.nanmean(corrs)
            group_annotators
            if label_name not in channel_dict:
                channel_dict[label_name] = {"channel": label_name, "annotator_corrs": [], "overall_corrs": [], "annotator_sums": [], "overall_sums": []}
            channel_dict[label_name]["annotator_corrs"].append(channel_annotator_corrs)
            channel_dict[label_name]["overall_corrs"].append(channel_overall_corrs)
            channel_dict[label_name]["annotator_sums"].append(channel_annotator_sums)
            channel_dict[label_name]["overall_sums"].append(channel_overall_sums)
    
    rows = []
    for channel_key in channel_dict:
        channel_row = channel_dict[channel_key]
        annotator_sums = np.nanmean(channel_row["annotator_sums"], axis=0)
        annotator_corrs = np.nanmean(channel_row["annotator_corrs"], axis=0)
        overall_corr = np.nanmean(channel_row["overall_corrs"], axis=0)
        overall_sum = np.nanmean(channel_row["overall_sums"], axis=0)
        row = {"channel": channel_row["channel"], "mean corr": overall_corr, "mean occurrence": overall_sum}
    
        for annotator, corr in zip(unique_annotators, annotator_corrs.tolist()):
            row[f"{annotator} corr with group"] = corr
        
        for annotator, percentage in zip(unique_annotators, annotator_sums.tolist()):
            row[f"{annotator} deviation (%)"] = percentage
    
        rows.append(row)
    
    dif_table = pd.DataFrame(rows).sort_values("mean corr").reset_index().drop(columns=["index"])
    overall_row = dif_table.mean(numeric_only=True)
    overall_row["channel"] = "overall"
    dif_table = pd.concat([dif_table, pd.DataFrame(overall_row).T]).reset_index().drop(columns=["index"])
    
    dirpath = iot.output_csvs_path() / "annotator_analysis"
    iot.create_missing_folder_recursive(dirpath)
    fullpath = dirpath / "channel_difs.csv"
    dif_table.to_csv(fullpath, sep="\t")
    annotator_names = "-".join(unique_annotators)
    df_len = len(dif_table)
    dl.write_to_manifest_new_summary_file(
            dl.STATISTICS_TYPE,
            fullpath,
            f"annotators:{annotator_names}",
            f"{df_len}",
        )

def run_annotator_dif_analysis(tasks=["task5"], color_k = 2):
    samples = dfs.get_sample_dataframe()
    samples = dfs.select_for_task(samples, tasks=nt.task_names_no_prefix(tasks))
    anchor_samples, unique_groups, annotator_dict, unique_annotators = get_groups_and_annotators(samples, color_k=color_k)
    create_figures(anchor_samples, unique_groups, annotator_dict, color_k=color_k)
    create_table(anchor_samples, unique_groups, annotator_dict, unique_annotators)
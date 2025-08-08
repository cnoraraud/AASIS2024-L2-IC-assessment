import io_tools as iot
import pandas as pd
import os
import numpy as np
import data_logger as dl
import random
import naming_tools as nt
import dataframe_sourcer as dfs
import dataframe_reader as dfr
import summary_reader as sumr

KEY_COLS = ["channel_1", "channel_2", "feature"]

CONSTRUCT = [
    ["mean_segment_density","self (ann.) ff:yaw",""],
    ["total_segment_width","self (ann.) hand:fidget",""],
    ["mean_times_relative","self (ext.) ic:turn","self (ann.) hand:gesture"],
    ["mean_segment_delay_count","self (ext.) ic:turn","self (ann.) hand:gesture"],
    ["total_segment_width","self (ann.) body:back",""],
    ["mean_segment_overlap_ratio","other (ann.) body:back","self (ann.) body:back"],
    ["mean_segment_overlap_ratio","other (ann.) body:forward","self (ann.) body:forward"],
    ["mean_segment_delay_count","other (ext.) ic:turn","self (ann.) text:laughing"],
    ["count_delay","other (ext.) ic:turn","self (ann.) ff:surprise"],
    ["count_delay","other (ext.) ic:turn","self (ann.) ff:disgust"],
    ["count_delay","other (ext.) ic:turn","self (ann.) ff:happiness"],
    ["mean_segment_delay_count","other (ext.) ic:turn","self (ann.) ff:surprise"],
    ["mean_segment_delay_count","other (ext.) ic:turn","self (ann.) head:nodding"],
    ["count_delay","other (ext.) ic:turn","self (ann.) text:bc"],
    ["mean_segment_width","self (ann.) text:bc",""],
    ["mean_delay","[all] (ann.) speechoverlap","self (ext.) vad"],
    ["mean_delay","self (ann.) text:paral","self (ext.) vad"],
    ["mean_delay","other (ann.) text:paral","self (ext.) vad"],
    ["mean_delay","self (ann.) text:hesitation","self (ext.) vad"],
    ["mean_delay","other (ann.) text:hesitation","self (ext.) vad"],
    ["mean_segment_width","self (ext.) ic:turn",""],
    ["mean_segment_overlap_count","self (ext.) ic:turn","self (ann.) text:name"],
    ["mean_segment_overlap_count","self (ext.) ic:turn","self (ann.) pause"],
    ["mean_times_from_end","self (ext.) ic:turn","self (ann.) text:name"],
    ["mean_times_from_end","self (ext.) ic:turn","self (ann.) pause"],
    ["mean_delay","other (ann.) text:name","self (ann.) text:text"],
    ["mean_delay","other (ann.) pause","self (ann.) text:text"],
    ["mean_delay","self (ann.) text:bc","self (ext.) ic:turn"],
    ["mean_delay","other (ext.) text:text","self (ext.) text:text"],
    ["segment_count","self (ext.) ic:turn",""],
    ["segment_count","self (ann.) text:bc",""]
]

def load_csv(test_dir, selected_group, csv_to_collate):
    csv_path = iot.output_csvs_path() / test_dir / selected_group / csv_to_collate
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path, sep="\t")
    first_column_is_index = np.all(df.index == df[df.columns[0]].values)
    
    if first_column_is_index:
        df = df.drop(columns=[df.columns[0]])
    label_df = df["labels"].str.split("|", expand=True, regex=False)
    df = df.drop(columns=["labels"])

    next_column = 0
    
    channel_1_value = ""
    if len(label_df.columns) > 0:
        label_df[0] = label_df[0].str.strip()
        channel_1_value = label_df[0]
    df.insert(next_column, "channel_1", channel_1_value)
    next_column += 1

    channel_2_value = ""
    if len(label_df.columns) > 1:
        label_df[1] = label_df[1].str.strip()
        channel_2_value = label_df[1]
    df.insert(next_column, "channel_2", channel_2_value)
    next_column += 1

    feature = "unknown_feature"
    for potential_feature in nt.ALL_FEATURES_LIST:
        if csv_to_collate.startswith(potential_feature):
            feature = potential_feature
            break
    df.insert(next_column, "feature", feature)
    next_column += 1

    test = csv_to_collate.replace(feature, "").split("test")[0].replace("_", "")
    df.insert(next_column, "test", test)
    next_column += 1
    
    file_path = "/".join([test_dir, selected_group, csv_to_collate])
    df.insert(next_column, "file_path", file_path)
    next_column += 1

    return df

def get_csv_list(test_dir, group_keywords, random_choice=True):
    group_paths = os.listdir(iot.output_csvs_path() / test_dir)
    potential_groups = []
    for group_path in group_paths:
        if not isinstance(group_keywords, type(None)):
            all_group_keywords_found = True
            for group_keyword in group_keywords:
                all_group_keywords_found = all_group_keywords_found and (group_keyword in str(group_path))
            if not all_group_keywords_found:
                continue
        potential_groups.append(group_path)
    if len(potential_groups) == 0:
        return [], None
    elif random_choice:
        selected_group = random.choice(potential_groups)
    else:
        selected_group = potential_groups[-1]
    csvs_to_collate = os.listdir(iot.output_csvs_path() / test_dir / selected_group)
    return csvs_to_collate, selected_group

def get_test_collation(test_dir, group_keywords, random_choice=True):
    dfs = []
    csvs_to_collate, selected_group = get_csv_list(test_dir, group_keywords, random_choice=random_choice)
    for csv_to_collate in csvs_to_collate:
        df = load_csv(test_dir, selected_group, csv_to_collate)
        if (not isinstance(df, type(None))) and (len(df) > 0):
            df = df.dropna(axis=1, how='all')
            dfs.append(df)
    concat_df = None
    if len(dfs) > 0:
        concat_df = pd.concat(dfs, ignore_index=True)
    return concat_df

def get_collations(tasks=None, group_keywords=None, random_choice=True):
    if not isinstance(tasks, list):
        tasks = [tasks]
    if not isinstance(group_keywords, list):
        group_keywords = [group_keywords]
    
    output_csvs = os.listdir(iot.output_csvs_path())
    test_dirs = []
    for output_csv_path in output_csvs:
        if not isinstance(tasks, type(None)):
            taskstring = output_csv_path.split("(")[-1].replace(")","")
            if "master" in output_csv_path:
                continue
            all_tasks_found = True
            for task in nt.task_names_no_prefix(tasks):
                all_tasks_found = all_tasks_found and (task in taskstring)
            if not all_tasks_found:
                continue
        test_dirs.append(output_csv_path)

    test_collations = dict()
    for test_dir in test_dirs:
        test_collation = get_test_collation(test_dir, group_keywords=group_keywords, random_choice=random_choice)
        test_collations[test_dir] = test_collation
    return test_collations

def merge_to_master(collations):
    collation_keys = sorted(list(collations.keys()))
    master_df = None
    for i in range(len(collation_keys)):
        current_key = collation_keys[i]
        new_df = collations[current_key]
        new_df[current_key] = True
        if i == 0:
            master_df = new_df
        else:
            prev_key = collation_keys[i-1]
            master_df = master_df.merge(new_df, on=KEY_COLS, how="outer", suffixes=(f" {prev_key}", f" {current_key}"))
    master_df = master_df.groupby(KEY_COLS).first().reset_index()
    test_cols = []
    for col in master_df.columns:
        if col.startswith("test") or col.startswith("Unnamed:"):
            test_cols.append(col)
    master_df = master_df.drop(columns=test_cols)
    return master_df

def merge_conds(conds, any=False):
    if any:
        return pd.concat(conds, axis=1).any(axis=1)
    return pd.concat(conds, axis=1).all(axis=1)
    
def filter_normality(df, keys, norm_p, norm_val):
    conds = []
    for key in keys:
        norm_p_conds = []
        for col in df.columns:
            if (not "N-p" in col) or (not key in col):
                continue
            norm_p_conds.append(df[col] <= norm_p)

        if len(norm_p_conds) == 0:
            continue
        norm_p_cond = merge_conds(norm_p_conds)
    
        for col in df.columns:
            if ((not "skew" in col) and (not "kurt" in col)) or (not key in col):
                continue
            not_normal = (df[col].abs() < norm_val) & norm_p_cond
            conds.append(~not_normal)

    normality_filter = merge_conds(conds)
    return df[normality_filter]

def filter_nobs(df, keys, nob_ratio, hard_bottom=0):
    conds = []
    for key in keys:
        for col in df.columns:
            if (not "nobs" in col) or (not key in col):
                continue
            max_nobs = df[col].max()
            desired_nobs = max(hard_bottom, round(max_nobs * nob_ratio))
            below_desired_nobs = df[col] < desired_nobs
            conds.append(~below_desired_nobs)
    nobs_filter = merge_conds(conds)
    return df[nobs_filter]

def filter_missing_required_columns(df, req_cols=["welch_ttest_p"]):
    conds = []
    for col in req_cols:
        required_column_missing = df[col].isnull()
        conds.append(~required_column_missing)
    required_filter = merge_conds(conds)
    return df[required_filter]

def filter_channels(df, chans=[], reverse=False):
    conds = []
    for chan in chans:
        contained_in_channel_1 = df["channel_1"].str.contains(chan, regex=False)
        contained_in_channel_2 = df["channel_2"].str.contains(chan, regex=False)
        contained_in_either_channel = contained_in_channel_1 | contained_in_channel_2
        if reverse:
            conds.append(contained_in_either_channel)
        else:
            conds.append(~contained_in_either_channel)
    channel_filter = merge_conds(conds)
    return df[channel_filter]

def filter_features(df, feats=[], reverse=False):
    conds = []
    for feat in feats:
        contained_in_feature = df["feature"].str.contains(feat, regex=False)
        if reverse:
            conds.append(contained_in_feature)
        else:
            conds.append(~contained_in_feature)
    feature_filter = merge_conds(conds)
    return df[feature_filter]

def filter_columns(df, bad_keys=["file_path", "Unnamed: 0"]):
    bad_cols = []
    for col in df.columns:
        for bad_key in bad_keys:
            if bad_key in col:
                bad_cols.append(col)
                continue
    return df.drop(columns=bad_cols)

def filter_construct(df):
    all_conds = []
    for c_row in CONSTRUCT:
        cond1 = df["feature"] == c_row[0]
        cond2 = df["channel_1"] == c_row[1]
        if len(c_row[2]) > 0:
            cond3 = df["channel_2"] == c_row[2]
            cond = cond1 & cond2 & cond3
        else:
            cond = cond1 & cond2
        if cond.sum() == 0:
            print(c_row)
        else:
            all_conds.append(cond)
    cond = merge_conds(all_conds, any=True)
    return df[cond]

def add_bonferroni_rejection(df, p_col, alpha=0.05):
    ps = df[p_col].sort_values()
    m = len(df)
    corrected_alpha = alpha/m
    ps_c = ps < corrected_alpha
    df[f"bonferroni rejection ({alpha})"] = ps_c
    dl.log(f"Bonferroni correction rejected {ps_c.sum()}. Corrected alpha: {corrected_alpha}")

def add_holm_rejection(df, p_col, alpha=0.05):
    ps = df[p_col].sort_values()
    m = len(df)
    best_k = 1
    for k in range(1, m+1):
        p = ps[k - 1]
        corrected_alpha = alpha/(m + 1 -k)
        if p > corrected_alpha:
            dl.log(f"Holm correction stopped at {best_k}. Bad alpha: {corrected_alpha}")
            break
        best_k = k
    ps_c = ps.astype("bool")
    ps_c[:] = False
    ps_c[0:best_k-1] = True
    df[f"holm rejection ({alpha})"] = ps_c

def add_hochberg_rejection(df, p_col, alpha=0.05):
    ps = df[p_col].sort_values()
    m = len(df)
    best_k = 1
    for k in range(1, m+1):
        p = ps[k - 1]
        corrected_alpha = alpha/(m + 1 -k)
        if p <= corrected_alpha:
            best_k = k
    ps_c = ps.astype("bool")
    ps_c[:] = False
    ps_c[0:best_k-1] = True
    dl.log(f"Hochberg correction stopped at {best_k}.")
    df[f"hochberg rejection ({alpha})"] = ps_c


ALL_FEATURES = "all_features"
VALID_FEATURES = "valid_features"
BASIC_VALID_FEATURES = "basic_valid_features"
CONSTRUCT_FEATURES = "construct_features"
CENTRAL_STATISTIC_NAMES = ["ag_p holistic_cefr", "rating_holistic_spearman_corr_p speakerid"]
def create_master_tables(tasks=None, group_keywords=None, central_statistic_names=CENTRAL_STATISTIC_NAMES):
    task_name = "".join(sorted(["5"]))
    
    collations = get_collations(tasks, group_keywords)
    master_df = merge_to_master(collations)
    master_df["channel_1"] = master_df["channel_1"].fillna("")
    master_df["channel_2"] = master_df["channel_2"].fillna("")
    master_df["feature"] = master_df["feature"].fillna("")
    
    for central_statistic_name in central_statistic_names:
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Creating master tables for task group ({task_name}), focusing on statistic {central_statistic_name}")
        central_statistic = f"{central_statistic_name}_({task_name})"
        
        filtered_df0 = master_df.copy()

        filtered_df = filtered_df0.copy()
        filtered_df = filter_missing_required_columns(filtered_df, req_cols=[central_statistic])
        filtered_df = filter_features(filtered_df, feats=["median","var", "distance"])
        filtered_df = filter_columns(filtered_df)
        
        filtered_df2 = filtered_df.copy()
        filtered_df2 = filter_channels(filtered_df2, chans=["cep:","ff:","energy"])

        filtered_df3 = filtered_df0.copy()
        filtered_df3 = filter_construct(filtered_df3)
        
        dfs = {ALL_FEATURES: filtered_df0, VALID_FEATURES: filtered_df, BASIC_VALID_FEATURES: filtered_df2, CONSTRUCT_FEATURES: filtered_df3}
        
        alpha=0.05
        for df_key in dfs:
            df = dfs[df_key]
            df = df.sort_values(by=central_statistic).reset_index().drop(columns=["index"])
            dl.log(f"Post-Hoc Statistics for: {df_key}")
            add_bonferroni_rejection(df, central_statistic, alpha=alpha)
            add_holm_rejection(df, central_statistic, alpha=alpha)
            add_hochberg_rejection(df, central_statistic, alpha=alpha)
        
            dfs[df_key] = df

        savepath = iot.output_csvs_path() / f"master_({task_name})"
        iot.create_missing_folder_recursive(savepath)
        for df_key in dfs:
            fullpath = savepath / f"{df_key}_({central_statistic_name}).csv"
            df = dfs[df_key]
            df.to_csv(fullpath, sep='\t')
            df_n = len(df)
            dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"task_group_{task_name}:{len(master_df)}", f"{df_n}")

def get_y_column(rating):
    return f"y_{rating}"
    
def get_x_column(chan1, chan2, feat):
    return f"x_{feat} >> {chan1}) | {chan2}"

def get_speaker_data(df, rating_fields, tasks, sample_groupings=None, summaries=None):
    if isinstance(sample_groupings, type(None)):
        speakers, samples = dfs.get_speakers_samples_full_dataframe(tasks)
        speaker_group = dfr.get_speaker_groups(speakers, ["SpeakerID"])
        sample_groupings = dfr.find_sample_groupings(speaker_group, samples)
    if isinstance(summaries, type(None)):
        summaries = sumr.load_summaries(tasks=tasks)
    x_columns_to_fields = dict()
    x_columns = []
    y_columns = []
    for i, key_fields in df[KEY_COLS].iterrows():
        feat = key_fields["feature"]
        chan1 = key_fields["channel_1"]
        chan2 = key_fields["channel_2"]
        x_column = get_x_column(chan1, chan2, feat)
        x_columns.append(x_column)
        x_columns_to_fields[x_column] = {"feature": feat, "channel_1": chan1, "channel_2": chan2}
    for rating in rating_fields:
        y_column = get_y_column(rating)
        y_columns.append(y_column)
    
    speaker_data = []
    for group_key in sample_groupings.keys():
        samples = sample_groupings[group_key]
        for sample_key in samples:
            sample = samples[sample_key]
            speaker_id = sample["speaker"]
            S = sample["S"]
            npy = sample["npy"]
            ratings = sample["ratings"]
            speaker_row = {"speaker": speaker_id}
            for rating, y_column in zip(rating_fields, y_columns):
                speaker_row[y_column] = ratings[rating]
    
            performance_length = sumr.get(summaries, f"{npy}/summary/general/performance_length")
            label_summaries = sumr.get(summaries, f"{npy}/summary/label_summaries")
            if isinstance(label_summaries, type(None)):
                dl.log(f"No label summaries were found for {npy}")
            for i, x_column in enumerate(x_columns):
                x_fields = x_columns_to_fields[x_column]
                feat = x_fields["feature"]
                feature_type, normalizable = nt.get_feature_type(feat)
                chan1 = x_fields["channel_1"]
                chan2 = x_fields["channel_2"]
                name = f"{feat} > {chan1} | {chan2}"
                l1 = nt.get_S_label(chan1, S)
                l2 = nt.get_S_label(chan2, S)
                val = np.nan
                if feature_type == nt.PROACTIVE:
                    val = sumr.get(label_summaries, f"{l1}/self/{feat}", fallback=np.nan)
                if feature_type == nt.REACTIVE:
                    val = sumr.get(label_summaries, f"{l1}/others/{l2}/{feat}", fallback=np.nan)
                if normalizable == nt.NORMALIZABLE:
                    val = val / performance_length
                x_column = get_x_column(chan1, chan2, feat)
                
                speaker_row[x_column] = val
            speaker_data.append(speaker_row)
    return pd.DataFrame(speaker_data), x_columns, y_columns
import summary_reader as sumr
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import io_tools as iot
import data_displayer as dd

def get_groups(sample_groupings, feature_name, self_labels=None, other_labels=None, collapse=False, sample_size=None, feature_type=None, self_features=None, other_features=None, general_features=None):
    self_labels, other_labels = sumr.find_noned_labels(sample_groupings, self_labels=self_labels, other_labels=other_labels)
    self_features, other_features, general_features = sumr.find_noned_features(sample_groupings, self_features=self_features, other_features=other_features, general_features=general_features)
    sample_size, feature_type = sumr.find_sample_size_and_feature_type(sample_groupings, feature_name, self_features=self_features, other_features=other_features, general_features=general_features)

    results = dict()
    for group_name in sample_groupings:
        grouping = sample_groupings[group_name]
        M, ML0, ML1, ML2 = sumr.get_data_matrix(grouping, feature_name, collapse=collapse, feature_type=feature_type, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
        results[group_name] = {"name": group_name, "matrix": M, "sessions": ML0, "self_labels": ML1, "other_labels": ML2}
    return results

def normalize_groupings(results, performance_results):
    normalized_results = dict()
    for group_key in results:
        res = results[group_key]
        perf_res = performance_results[group_key]
        norm_M = res["matrix"] / perf_res["matrix"]
        normalized_results[group_key] = {"name": res["name"], "matrix": norm_M, "sessions": res["sessions"], "self_labels": res["self_labels"], "other_labels": res["other_labels"]}
    return normalized_results

def check_equality(els):
    for el1 in els:
        for el2 in els:
            if not np.all(el1 == el2):
                return False
    return True

def get_medians(L1, L2, Ms, names, properties = {}):
    labels = []
    for l in L1:
        labels.append(l.split(" ")[-1])
    obj = {"labels": labels}
    for M, name in zip(Ms, names):
        obj[name] = np.nanmedian(M, axis=0).flatten().tolist()
    return pd.DataFrame(obj)

def get_ttest_for_binary(L1, L2, Ms, names, properties = {}):
    obj = {"labels": [], "statistic": [], "p_value": []}
    for row in range(len(L1)):
        samples = []
        for M in Ms:
            sample = M[:, row, 0]
            sample = sample[~np.isnan(sample)]
            samples.append(sample)
        statistic, p_value = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        obj["labels"].append(L1[row])
        obj["statistic"].append(statistic)
        obj["p_value"].append(p_value)
    return pd.DataFrame(obj)

def get_ttest_for_binary_relational(L1, L2, Ms, names, properties = {}):
    obj = {"labels": [], "statistic": [], "p_value": []}
    for row in range(len(L1)):
        for col in range(len(L2)):
            samples = []
            for M in Ms:
                sample = M[:, row, col]
                sample = sample[~np.isnan(sample)]
                samples.append(sample)
            statistic, p_value = stats.ttest_ind(samples[0], samples[1], equal_var=False)
            obj["labels"].append(f"{L1[row]} + {L2[col]}")
            obj["statistic"].append(statistic)
            obj["p_value"].append(p_value)
    return pd.DataFrame(obj)

def compare_groupings(results, method, properties = {}):
    Ms = []
    L1s = []
    L2s = []
    Ns = []
    names = []
    for res_key in sorted(list(results.keys())):
        res = results[res_key]
        name = res["name"]
        M = res["matrix"]
        L1 = res["self_labels"]
        L2 = res["other_labels"]
        N = len(res["sessions"])
        Ms.append(M)
        L1s.append(L1)
        L2s.append(L2)
        Ns.append(N)
        names.append(name)
    
    if not check_equality(L1s) or not check_equality(L2s):
        print("Labels don't match")

    L1 = L1s[0]
    L2 = L2s[0]
    return method(L1, L2, Ms, names, properties)

def ttest_self_features(res, collapse, self_labels, other_labels, self_features, other_features, general_features, performance_results):
    feature_set = ['mean_segment_density', 'mean_segment_width', 'segment_count', 'std_segment_density', 'std_segment_width', 'total_segment_mass', 'total_segment_width', 'var_segment_density', 'var_segment_width']
    normalizable_features = ['segment_count', 'total_segment_mass', 'total_segment_width']
    for feature_name in feature_set:
        feature_results = get_groups(res, feature_name, collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
        normalize = feature_name in normalizable_features
        if normalize:
            feature_results = normalize_groupings(feature_results, performance_results)
        ttest = compare_groupings(feature_results, get_ttest_for_binary)
        ttest.dropna().sort_values("p_value").reset_index().drop(columns = ["index"]).to_csv(iot.output_csvs_path() / f"{feature_name}_welchs_ttest.csv", sep='\t')

def ttest_relational_features(res, collapse, self_labels, other_labels, self_features, other_features, general_features, performance_results):
    feature_set = ['count_delay', 'count_overlap', 'mean_delay', 'mean_overlap', 'mean_segment_delay_count', 'mean_segment_overlap_count', 'mean_segment_overlap_ratio', 'median_delay', 'median_overlap', 'median_segment_delay_count', 'median_segment_overlap_count', 'median_segment_overlap_ratio', 'pearson_corr', 'spearman_corr', 'total_overlap']
    normalizable_features = ['count_delay', 'count_overlap', 'total_overlap']
    for feature_name in feature_set:
        feature_results = get_groups(res, feature_name, collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
        normalize = feature_name in normalizable_features
        if normalize:
            feature_results = normalize_groupings(feature_results, performance_results)
        keys = list(feature_results.keys())

        # Produce Dataframes
        ttest = compare_groupings(feature_results, get_ttest_for_binary_relational)
        split_name = "-".join(keys)
        ttest.dropna().sort_values("p_value").reset_index().drop(columns = ["index"]).to_csv(iot.output_csvs_path() / f"{feature_name}_welchs_ttest_{split_name}.csv", sep='\t')
        try:
            # Produce Figures
            feature_results_1 = feature_results[keys[0]]
            feature_results_2 = feature_results[keys[1]]
            data1 = np.nanmedian(feature_results_1["matrix"], axis=0)
            data2 = np.nanmedian(feature_results_2["matrix"], axis=0)
            
            width = 20
            height = 20
            if collapse:
                height=10
            dd.r(data1, f"Median {feature_name} with collapsed speakers for \'{keys[0]}\'", feature_results_1["self_labels"], feature_results_1["other_labels"], width=width, height=height, savefig=True, colorbar=True)
            dd.r(data2, f"Median {feature_name} with collapsed speakers for \'{keys[1]}\'", feature_results_2["self_labels"], feature_results_2["other_labels"], width=width, height=height, savefig=True, colorbar=True)
            dd.r(data2-data1, f"Median {feature_name} with collapsed speakers for difference between\'{keys[1]}\' and \'{keys[0]}\'", feature_results_2["self_labels"], feature_results_2["other_labels"], width=width, height=height, savefig=True, colorbar=True)
        except:
            print(f"Failed {feature_name}")

def analysis_precompute(res, collapse=True):
    self_labels, other_labels = sumr.find_noned_labels(res)
    self_features, other_features, general_features = sumr.find_noned_features(res)
    performance_results = get_groups(res, "performance_length", collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
    return collapse, self_labels, other_labels, self_features, other_features, general_features, performance_results
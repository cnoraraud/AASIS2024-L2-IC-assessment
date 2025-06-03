import summary_reader as sumr
import numpy as np
import numpy_wrapper as npw
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

def create_statistics_dict(N1, N2):
    obj = {"labels": [],
        "statistic": [],
        "p_value": [],
        f"nobs {N1}": [],
        f"nobs {N2}": [],
        f"median {N1}": [],
        f"median {N2}": [],
        f"mean {N1}": [],
        f"mean {N2}": [],
        f"var {N1}": [],
        f"var {N2}": [],
        f"skew {N1}": [],
        f"skew {N2}": [],
        f"kurt {N1}": [],
        f"kurt {N2}": [],
        f"N-p {N1}": [],
        f"N-p {N2}": []}
    return obj

def add_sample_comparison_to_statics_dict(obj, samples):
    statistic = np.nan
    p_value = np.nan

    if npw.valid(samples[0], 4) and npw.valid(samples[1], 4):
        statistic, p_value = stats.ttest_ind(samples[0], samples[1], equal_var=False)
    obj["statistic"].append(statistic)
    obj["p_value"].append(p_value)

def add_sample_analysis_to_statistics_dict(obj, sample, N):
    rng = np.random.default_rng()
    nobs = np.nan
    minmax = np.nan
    mean = np.nan
    var = np.nan
    skew = np.nan
    kurt = np.nan
    gof_pvalue = np.nan
    median = np.nan

    if npw.valid(sample, 4) & npw.valid_dist(sample):
        nobs, minmax, mean, var, skew, kurt = stats.describe(sample)
        gof = stats.goodness_of_fit(stats.norm, sample, statistic='ad', rng=rng)
        gof_pvalue = gof.pvalue
        median = np.median(sample)

    obj[f"nobs {N}"].append(nobs)
    obj[f"median {N}"].append(median)
    obj[f"mean {N}"].append(mean)
    obj[f"var {N}"].append(var)
    obj[f"skew {N}"].append(skew)
    obj[f"kurt {N}"].append(kurt)
    obj[f"N-p {N}"].append(gof_pvalue)
    

def get_ttest_for_binary(L1, L2, Ms, names, properties = {}):
    N1 = names[0]
    N2 = names[1]
    obj = create_statistics_dict(N1, N2)
    for row in range(len(L1)):
        samples = []
        for M in Ms:
            sample = M[:, row, 0]
            sample = sample[~np.isnan(sample)]
            samples.append(sample)
        obj["labels"].append(L1[row])
        add_sample_comparison_to_statics_dict(obj, samples)
        add_sample_analysis_to_statistics_dict(obj, samples[0], N1)
        add_sample_analysis_to_statistics_dict(obj, samples[1], N2)
    return pd.DataFrame(obj)

def get_ttest_for_binary_relational(L1, L2, Ms, names, properties = {}):
    N1 = names[0]
    N2 = names[1]
    obj = create_statistics_dict(N1, N2)
    for row in range(len(L1)):
        for col in range(len(L2)):
            samples = []
            for M in Ms:
                sample = M[:, row, col]
                sample = sample[~np.isnan(sample)]
                samples.append(sample)
            obj["labels"].append(f"{L1[row] : L2[col]}")
            add_sample_comparison_to_statics_dict(obj, samples)
            add_sample_analysis_to_statistics_dict(obj, samples[0], N1)
            add_sample_analysis_to_statistics_dict(obj, samples[1], N2)
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
        keys = list(feature_results.keys())
        split_name = "-".join(keys).lower()
        savepath = iot.output_csvs_path() / f"{split_name}"
        iot.create_missing_folder(savepath)

        ttest = compare_groupings(feature_results, get_ttest_for_binary)
        ttest.dropna().sort_values("p_value").reset_index().drop(columns = ["index"]).to_csv(savepath / f"{feature_name}_welchs_ttest_{split_name}.csv", sep='\t')

def ttest_relational_features(res, collapse, self_labels, other_labels, self_features, other_features, general_features, performance_results):
    feature_set = ['count_delay', 'count_overlap', 'mean_delay', 'mean_overlap', 'mean_segment_delay_count', 'mean_segment_overlap_count', 'mean_segment_overlap_ratio', 'median_delay', 'median_overlap', 'median_segment_delay_count', 'median_segment_overlap_count', 'median_segment_overlap_ratio', 'pearson_corr', 'spearman_corr', 'total_overlap']
    normalizable_features = ['count_delay', 'count_overlap', 'total_overlap']
    for feature_name in feature_set:
        feature_results = get_groups(res, feature_name, collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
        normalize = feature_name in normalizable_features
        if normalize:
            feature_results = normalize_groupings(feature_results, performance_results)
        keys = list(feature_results.keys())
        split_name = "-".join(keys).lower()
        savepath = iot.output_csvs_path() / f"{split_name}"
        iot.create_missing_folder(savepath)

        # Produce Dataframes
        ttest = compare_groupings(feature_results, get_ttest_for_binary_relational)
        ttest.dropna().sort_values("p_value").reset_index().drop(columns = ["index"]).to_csv(savepath / f"{feature_name}_welchs_ttest_{split_name}.csv", sep='\t')
        try:
            # Produce Figures
            feature_results_1 = feature_results[keys[0]]
            feature_results_2 = feature_results[keys[1]]
            data1 = np.nanmedian(feature_results_1["matrix"], axis=0)
            data2 = np.nanmedian(feature_results_2["matrix"], axis=0)
            
            width = 20
            height = 20
            collapse_string = ""
            if collapse:
                height=10
                collapse_string = "with collapsed features for "
            dd.r(data1, f"Median {feature_name} {collapse_string}\'{keys[0].lower()}\'", feature_results_1["self_labels"], feature_results_1["other_labels"], width=width, height=height, savefig=True, colorbar=True, subfolder=split_name)
            dd.r(data2, f"Median {feature_name} {collapse_string}\'{keys[1].lower()}\'", feature_results_2["self_labels"], feature_results_2["other_labels"], width=width, height=height, savefig=True, colorbar=True, subfolder=split_name)
            dd.r(data2-data1, f"Median {feature_name} {collapse_string}difference between\'{keys[1].lower()}\' and \'{keys[0].lower()}\'", feature_results_2["self_labels"], feature_results_2["other_labels"], width=width, height=height, savefig=True, colorbar=True, subfolder=split_name)
        except:
            print(f"Failed {feature_name}")

def analysis_precompute(res, collapse=True):
    self_labels, other_labels = sumr.find_noned_labels(res)
    self_features, other_features, general_features = sumr.find_noned_features(res)
    performance_results = get_groups(res, "performance_length", collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
    return collapse, self_labels, other_labels, self_features, other_features, general_features, performance_results
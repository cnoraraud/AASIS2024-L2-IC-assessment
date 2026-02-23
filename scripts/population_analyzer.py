import numpy as np
import numpy_wrapper as npw
from scipy import stats
import pandas as pd
from functools import reduce
import data_logger as dl
import naming_tools as nt


def get_label_name(l1=None, l2=None):
    l1_exists = not isinstance(l1, type(None))
    l2_exists = not isinstance(l2, type(None))
    if l1_exists and l2_exists:
        return f"{l1} | {l2}"
    if l1_exists:
        return f"{l1}"
    if l2_exists:
        return f"{l2}"
    return ""


def check_equality(els):
    for el1 in els:
        for el2 in els:
            if el1.shape != el2.shape:
                return False
            if not np.all(el1 == el2):
                return False
    return True


def find_total_intersection(els):
    return reduce(np.intersect1d, els)


def find_total_union(els):
    return reduce(np.union1d, els)


def fix_labels(label_list):
    if len(label_list) == 0:
        dl.log("Empty label list")
        return []

    intersection = find_total_intersection(label_list)
    union = find_total_union(label_list)
    missing = np.setdiff1d(union, intersection)

    if not check_equality(label_list):
        dl.log("Labels don't match")
        missing_labels = ", ".join(missing)
        dl.log(f"Labels {missing_labels} are sometimes missing.")

    return intersection


def base_dict():
    return {"labels": []}


def add_welchttest_dict(obj):
    obj["welch_ttest"] = []
    obj["welch_ttest_p"] = []


def add_description_dict(obj, names):
    for group_name in names:
        obj[f"nobs {group_name}"] = []
        obj[f"median {group_name}"] = []
        obj[f"mean {group_name}"] = []
        obj[f"var {group_name}"] = []
        obj[f"skew {group_name}"] = []
        obj[f"kurt {group_name}"] = []
        obj[f"N-p {group_name}"] = []


def add_sample_welchttest_to_dict(obj, samples):
    statistic = np.nan
    p_value = np.nan

    if npw.valid_dist(samples[0]) and npw.valid_dist(samples[1]):
        statistic, p_value = stats.ttest_ind(
            samples[0], samples[1], equal_var=False, nan_policy="omit"
        )
    obj["welch_ttest"].append(statistic)
    obj["welch_ttest_p"].append(p_value)


def add_sample_descriptions_to_dict(obj, samples, names):
    rng = np.random.default_rng()
    for sample, group_name in zip(samples, names):
        nobs = np.nan
        minmax = np.nan
        mean = np.nan
        var = np.nan
        skew = np.nan
        kurt = np.nan
        gof_pvalue = np.nan
        median = np.nan

        if npw.valid_dist(sample):
            nobs, minmax, mean, var, skew, kurt = stats.describe(
                sample, nan_policy="omit"
            )
            gof = stats.goodness_of_fit(stats.norm, sample, statistic="ad", rng=rng)
            gof_pvalue = gof.pvalue
            median = np.nanmedian(sample)

        obj[f"nobs {group_name}"].append(nobs)
        obj[f"median {group_name}"].append(median)
        obj[f"mean {group_name}"].append(mean)
        obj[f"var {group_name}"].append(var)
        obj[f"skew {group_name}"].append(skew)
        obj[f"kurt {group_name}"].append(kurt)
        obj[f"N-p {group_name}"].append(gof_pvalue)


def add_corrs_dict(obj, rating_names):
    for rating_name in rating_names:
        obj[f"{rating_name}_pearson_corr"] = []
        obj[f"{rating_name}_pearson_corr_p"] = []
        obj[f"{rating_name}_spearman_corr"] = []
        obj[f"{rating_name}_spearman_corr_p"] = []


def add_sample_corrs_to_dict(obj, samples, ratings, rating_names):
    for rating_name in rating_names:
        pearson_corr = np.nan
        pearson_corr_p = np.nan
        spearman_corr = np.nan
        spearman_corr_p = np.nan
        if rating_name in ratings:
            rating_samples = ratings[rating_name]
            notnan = ~np.isnan(rating_samples) & ~np.isnan(samples)
            if npw.valid_dist(rating_samples[notnan]) and npw.valid_dist(
                samples[notnan]
            ):
                spearman_res = stats.spearmanr(rating_samples[notnan], samples[notnan])
                pearson_res = stats.pearsonr(rating_samples[notnan], samples[notnan])

                pearson_corr = pearson_res.statistic
                pearson_corr_p = pearson_res.pvalue
                spearman_corr = spearman_res.statistic
                spearman_corr_p = spearman_res.pvalue

        obj[f"{rating_name}_pearson_corr"].append(pearson_corr)
        obj[f"{rating_name}_pearson_corr_p"].append(pearson_corr_p)
        obj[f"{rating_name}_spearman_corr"].append(spearman_corr)
        obj[f"{rating_name}_spearman_corr_p"].append(spearman_corr_p)


def add_anova_dict(obj):
    obj["anova"] = []
    obj["anova_p"] = []
    obj["anova_valid"] = []
    obj["ag"] = []
    obj["ag_p"] = []
    obj["ag_valid"] = []
    obj["kruskal"] = []
    obj["kruskal_p"] = []
    obj["kruskal_valid"] = []


def add_sample_anova_to_dict(obj, samples):
    anova_statistic = np.nan
    anova_p_value = np.nan
    ag_statistic = np.nan
    ag_p_value = np.nan
    kruskal_statistic = np.nan
    kruskal_p_value = np.nan

    anova_valid_samples = []
    ag_valid_samples = []
    kruskal_valid_samples = []
    for sample in samples:
        is_valid_dist_for_anova = npw.valid_dist(sample, total_n=2, unique_n=2)
        is_valid_dist_for_ag = npw.valid_dist(sample, total_n=2, unique_n=2)
        is_valid_dist_for_kruskal = npw.valid_dist(sample, total_n=5, unique_n=2)
        if is_valid_dist_for_anova:
            anova_valid_samples.append(sample)
        if is_valid_dist_for_ag:
            ag_valid_samples.append(sample)
        if is_valid_dist_for_kruskal:
            kruskal_valid_samples.append(sample)

    if len(anova_valid_samples) >= 2:
        anova = stats.f_oneway(*anova_valid_samples, nan_policy="omit")
        anova_statistic = anova.statistic
        anova_p_value = anova.pvalue
    if len(ag_valid_samples) >= 2:
        ag = stats.alexandergovern(*ag_valid_samples, nan_policy="omit")
        ag_statistic = ag.statistic
        ag_p_value = ag.pvalue
    if len(kruskal_valid_samples) >= 2:
        try:
            kruskal = stats.kruskal(*kruskal_valid_samples, nan_policy="omit")
            kruskal_statistic = kruskal.statistic
            kruskal_p_value = kruskal.pvalue
        except ValueError:
            dl.log("Attempted kruskal on uniform samples")

    obj["anova"].append(anova_statistic)
    obj["anova_p"].append(anova_p_value)
    obj["anova_valid"].append(len(anova_valid_samples))
    obj["ag"].append(ag_statistic)
    obj["ag_p"].append(ag_p_value)
    obj["ag_valid"].append(len(ag_valid_samples))
    obj["kruskal"].append(kruskal_statistic)
    obj["kruskal_p"].append(kruskal_p_value)
    obj["kruskal_valid"].append(len(kruskal_valid_samples))


def pairwise_comparison_proactive(L1, L2, Ms, names, ratings, properties=None):
    obj = base_dict()
    add_welchttest_dict(obj)
    add_description_dict(obj, names)
    for row in range(len(L1)):
        samples = []
        for M in Ms:
            sample = M[:, row, 0]
            sample = sample[~np.isnan(sample)]
            samples.append(sample)
        obj["labels"].append(L1[row])
        add_sample_welchttest_to_dict(obj, samples)
        add_sample_descriptions_to_dict(obj, samples, names)
    return dataframe_object(obj, "welch_ttest_p")


def pairwise_comparison_reactive(L1, L2, Ms, names, ratings, properties=None):
    obj = base_dict()
    add_welchttest_dict(obj)
    add_description_dict(obj, names)
    for row in range(len(L1)):
        for col in range(len(L2)):
            samples = []
            for M in Ms:
                sample = M[:, row, col]
                sample = sample[~np.isnan(sample)]
                samples.append(sample)
            obj["labels"].append(get_label_name(L1[row], L2[col]))
            add_sample_welchttest_to_dict(obj, samples)
            add_sample_descriptions_to_dict(obj, samples, names)
    return dataframe_object(obj, "welch_ttest_p")


def intragroup_comparison_proactive(L1, L2, Ms, names, ratings, properties=None):
    rating_names = properties["rating_names"]
    obj = base_dict()
    add_corrs_dict(obj, rating_names)
    for row in range(len(L1)):
        samples = []
        i = 0
        for M in Ms:
            i += 1
            sample = M[:, row, 0]
            sample = sample[~np.isnan(sample)]
            if np.size(sample) > 0:
                samples.append(np.nanmean(sample))
            else:
                samples.append(np.nan)
        obj["labels"].append(get_label_name(L1[row]))
        add_sample_corrs_to_dict(obj, np.array(samples), ratings, rating_names)
    return dataframe_object(obj)


def intragroup_comparison_reactive(L1, L2, Ms, names, ratings, properties=None):
    rating_names = properties["rating_names"]
    obj = base_dict()
    add_corrs_dict(obj, rating_names)
    for row in range(len(L1)):
        for col in range(len(L2)):
            samples = []
            for M in Ms:
                sample = M[:, row, col]
                sample = sample[~np.isnan(sample)]
                if np.size(sample) > 0:
                    samples.append(np.nanmean(sample))
                else:
                    samples.append(np.nan)
            obj["labels"].append(get_label_name(L1[row], L2[col]))
            add_sample_corrs_to_dict(obj, np.array(samples), ratings, rating_names)
    return dataframe_object(obj)


def nwise_comparison_proactive(L1, L2, Ms, names, ratings, properties=None):
    obj = base_dict()
    add_anova_dict(obj)
    add_description_dict(obj, names)
    for row in range(len(L1)):
        samples = []
        for M in Ms:
            sample = M[:, row, 0]
            sample = sample[~np.isnan(sample)]
            samples.append(sample)
        obj["labels"].append(get_label_name(L1[row]))
        add_sample_anova_to_dict(obj, samples)
        add_sample_descriptions_to_dict(obj, samples, names)
    return dataframe_object(obj, "anova_p")


def nwise_comparison_reactive(L1, L2, Ms, names, ratings, properties=None):
    obj = base_dict()
    add_anova_dict(obj)
    add_description_dict(obj, names)
    for row in range(len(L1)):
        for col in range(len(L2)):
            samples = []
            for M in Ms:
                sample = M[:, row, col]
                sample = sample[~np.isnan(sample)]
                samples.append(sample)
            obj["labels"].append(get_label_name(L1[row], L2[col]))
            add_sample_anova_to_dict(obj, samples)
            add_sample_descriptions_to_dict(obj, samples, names)
    return dataframe_object(obj, "anova_p")


def dataframe_object(obj, sort_by=None):
    df = pd.DataFrame(obj).dropna(thresh=2)
    if not isinstance(sort_by, type(None)):
        df = df.sort_values(sort_by).reset_index().drop(columns=["index"])
    return df


def compare_groupings(results, method, properties=None):
    Ms = []
    L1s = []
    L2s = []
    Ns = []
    names = []
    ratings = dict()
    for res_key in sorted(list(results.keys())):
        res = results[res_key]
        N = 0
        if not isinstance(res["sessions"], type(None)):
            N = len(res["sessions"])
        if N == 0:
            continue
        L1 = res[nt.INIT_L_KEY]
        L2 = res[nt.RESP_L_KEY]
        L1s.append(L1)
        L2s.append(L2)

    L1c = fix_labels(L1s)
    L2c = fix_labels(L2s)
    if len(L1c) == 0 and len(L2c) == 0:
        return pd.DataFrame(dict())

    for res_key in sorted(list(results.keys())):
        res = results[res_key]
        N = 0
        if not isinstance(res["sessions"], type(None)):
            N = len(res["sessions"])
        if N == 0:
            continue
        name = res["name"]
        M = res["matrix"]
        L1 = res[nt.INIT_L_KEY]
        L2 = res[nt.RESP_L_KEY]

        M_mod = M[np.ix_(np.full(M.shape[0], True), np.isin(L1, L1c), np.isin(L2, L2c))]
        M_mod = np.reshape(M_mod, (M.shape[0], len(L1c), len(L2c)))
        Ms.append(M_mod)
        Ns.append(N)
        names.append(name)
        for field in res:
            if "rating" not in field:
                continue
            if field not in ratings:
                ratings[field] = []
            field_flat = res[field].flatten()
            field_flat = field_flat[~np.isnan(field_flat)]
            if np.size(field_flat) > 0:
                ratings[field].append(np.nanmean(field_flat))
            else:
                ratings[field].append(np.nan)
    for rating in ratings:
        ratings[rating] = np.array(ratings[rating])

    return method(L1c, L2c, Ms, names, ratings, properties)

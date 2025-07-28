import summary_reader as sumr
import numpy as np
import npy_reader as npyr
import io_tools as iot
import data_displayer as dd
import dataframe_sourcer as dfs
import dataframe_reader as dfr
import data_logger as dl
import naming_tools as nt
import summary_reader as sr
import population_analyzer as popa

def get_sample_aggregates(sample_groupings, feature_name, self_labels=None, other_labels=None, collapse=False, sample_size=None, feature_type=None, self_features=None, other_features=None, general_features=None):
    self_labels, other_labels = sumr.fill_labels(sample_groupings, self_labels=self_labels, other_labels=other_labels)
    self_features, other_features, general_features = sumr.fill_features(sample_groupings, self_features=self_features, other_features=other_features, general_features=general_features)
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

SELF_FEATURES_LIST = ['mean_segment_density', 'mean_segment_width', 'segment_count', 'std_segment_density', 'std_segment_width', 'total_segment_mass', 'total_segment_width', 'var_segment_density', 'var_segment_width']
RELATIONAL_FEATURES_LIST = ['mean_difference', 'mean_times_relative', 'mean_times_from_start', 'mean_times_from_end', 'median_difference', 'median_times_relative', 'median_times_from_start', 'median_times_from_end', 'std_overlap', 'std_delay', 'std_segment_overlap_count', 'std_segment_delay_count', 'std_segment_overlap_ratio', 'std_difference', 'std_times_relative', 'std_times_from_start', 'std_times_from_end', 'count_delay', 'count_overlap', 'mean_delay', 'mean_overlap', 'mean_segment_delay_count', 'mean_segment_overlap_count', 'mean_segment_overlap_ratio', 'median_delay', 'median_overlap', 'median_segment_delay_count', 'median_segment_overlap_count', 'median_segment_overlap_ratio', 'pearson_corr', 'spearman_corr', 'total_overlap']
NORMALIZABLE_FEATURES_LIST = ['segment_count', 'total_segment_mass', 'total_segment_width', 'count_delay', 'count_overlap', 'total_overlap']

COLLAPSE_TO_SELF_LIST = SELF_FEATURES_LIST + ['pearson_corr', 'spearman_corr']
COLLAPSE_TO_OTHER_LIST = ['mean_difference', 'mean_times_relative', 'mean_times_from_start', 'mean_times_from_end', 'median_difference', 'median_times_relative', 'median_times_from_start', 'median_times_from_end', 'std_overlap', 'std_delay', 'std_segment_overlap_count', 'std_segment_delay_count', 'std_segment_overlap_ratio', 'std_difference', 'std_times_relative', 'std_times_from_start', 'std_times_from_end', 'count_delay', 'count_overlap', 'mean_delay', "mean_overlap", "mean_segment_delay_count", "mean_segment_overlap_ratio", "median_delay", "median_overlap", "median_segment_delay_count", "median_segment_overlap_count", "median_segment_overlap_ratio", "total_overlap"]

def decide_collapse(collapse, feature_name):
    if not collapse:
        return sr.COLLAPSE_NONE
    if feature_name in COLLAPSE_TO_SELF_LIST:
        return sr.COLLAPSE_SELF_LABELS
    if feature_name in COLLAPSE_TO_OTHER_LIST:
        return sr.COLLAPSE_OTHER_LABELS

def add_ratings_to_results(res, ratings_results):
    for res_key in res:
        for ratings_key in ratings_results:
            res[res_key][ratings_key] = ratings_results[ratings_key][res_key]["matrix"]

def test(test_method, feature_set, res, collapse, self_labels, other_labels, self_features, other_features, general_features, ratings, performance_results, ratings_results, overwrite=True, grouping_name = "default"):
    test_type = test_method.__name__
    dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Testing \'{test_type}\' (Collapse:{collapse}, Overwrite:{overwrite})")
    for feature_name in feature_set:
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Doing statistics for \'{feature_name}\'")
        feature_collapse = decide_collapse(collapse, feature_name)
        feature_results = get_sample_aggregates(res, feature_name, collapse=feature_collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
        add_ratings_to_results(feature_results, ratings_results)
        normalize = feature_name in NORMALIZABLE_FEATURES_LIST
        if normalize:
            feature_results = normalize_groupings(feature_results, performance_results)
        
        keys = list(feature_results.keys())
        if len(keys) > 1:
            keys = [keys[0], keys[-1]]
        split_name = nt.sanitize_filename("-".join(keys)).lower()
        if collapse:
            split_name = f"{split_name}_collapsed"
        sanitized_feature_name = nt.sanitize_filename(feature_name).lower()
        sanitized_grouping_name = nt.sanitize_filename(grouping_name).lower()
        savepath = iot.output_csvs_path() / f"{sanitized_grouping_name}" / f"{split_name}"
        iot.create_missing_folder_recursive(savepath)
        
        try:
            test_method(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse)
        except Exception as e:
            dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Doing statistics failed for feature \'{feature_name}\'")
            dl.log_stack("Failure in running experiments")

    dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Finished test \'{test_type}\'")
    
# lots of repetion below, probably could optimize it better
def pairwise_test_self_features(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse):
    fullpath = savepath / f"{sanitized_feature_name}_welchs_ttest_{split_name}.csv"

    if not overwrite and fullpath.exists():
        existed_log = f"{fullpath} existed, skipping statistics step."
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, existed_log)
    else:
        ttest = popa.compare_groupings(feature_results, popa.pairwise_comparison_self)
        ttest.to_csv(fullpath, sep='\t')
        ttest_n = len(ttest)
        dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"{sanitized_feature_name}:{split_name}", f"{ttest_n}")
    
def pairwise_test_relational_features(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse):
    fullpath = savepath / f"{sanitized_feature_name}_welchs_ttest_{split_name}.csv"

    # Produce Dataframes
    if not overwrite and fullpath.exists():
        existed_log = f"{fullpath} existed, skipping statistics step."
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, existed_log)
    else:
        ttest = popa.compare_groupings(feature_results, popa.pairwise_comparison_relational)
        ttest.to_csv(fullpath, sep='\t')
        ttest_n = len(ttest)
        dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"{sanitized_feature_name}:{split_name}", f"{ttest_n}")
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
        figure_folder = f"{sanitized_grouping_name}/{split_name}"
        dd.r(data1, f"Median {sanitized_feature_name} {collapse_string}\'{keys[0].lower()}\'", feature_results_1["self_labels"], feature_results_1["other_labels"], width=width, height=height, savefig=True, colorbar=True, subfolder=figure_folder, overwrite=overwrite)
        dd.r(data2, f"Median {sanitized_feature_name} {collapse_string}\'{keys[1].lower()}\'", feature_results_2["self_labels"], feature_results_2["other_labels"], width=width, height=height, savefig=True, colorbar=True, subfolder=figure_folder, overwrite=overwrite)
        dd.r(data2-data1, f"Median {sanitized_feature_name} {collapse_string}difference between \'{keys[1].lower()}\' and \'{keys[0].lower()}\'", feature_results_2["self_labels"], feature_results_2["other_labels"], width=width, height=height, savefig=True, colorbar=True, subfolder=figure_folder, overwrite=overwrite)
    except:
        dl.log(f"Failed {sanitized_feature_name}")

SPEAKERWISE_PROPERTIES = {"rating_names": ["rating_holistic", "rating_IC_score", "rating_turn_taking"]}
def speakerwise_test_self(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse):
    fullpath = savepath / f"{sanitized_feature_name}_speakerwise_test_{split_name}.csv"

    if not overwrite and fullpath.exists():
        existed_log = f"{fullpath} existed, skipping statistics step."
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, existed_log)
    else:
        ttest = popa.compare_groupings(feature_results, popa.speakerwise_comparison_self, SPEAKERWISE_PROPERTIES)
        ttest.to_csv(fullpath, sep='\t')
        ttest_n = len(ttest)
        dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"{sanitized_feature_name}:{split_name}", f"{ttest_n}")

def speakerwise_test_relational(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse):
    fullpath = savepath / f"{sanitized_feature_name}_speakerwise_test_{split_name}.csv"

    if not overwrite and fullpath.exists():
        existed_log = f"{fullpath} existed, skipping statistics step."
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, existed_log)
    else:
        ttest = popa.compare_groupings(feature_results, popa.speakerwise_comparison_relational, SPEAKERWISE_PROPERTIES)
        ttest.to_csv(fullpath, sep='\t')
        ttest_n = len(ttest)
        dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"{sanitized_feature_name}:{split_name}", f"{ttest_n}")

def nwise_test_self(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse):
    fullpath = savepath / f"{sanitized_feature_name}_nwise_test_{split_name}.csv"

    if not overwrite and fullpath.exists():
        existed_log = f"{fullpath} existed, skipping statistics step."
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, existed_log)
    else:
        ttest = popa.compare_groupings(feature_results, popa.nwise_comparison_self)
        ttest.to_csv(fullpath, sep='\t')
        ttest_n = len(ttest)
        dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"{sanitized_feature_name}:{split_name}", f"{ttest_n}")

def nwise_test_relational(feature_results, keys, savepath, sanitized_feature_name, sanitized_grouping_name, split_name, overwrite, collapse):
    fullpath = savepath / f"{sanitized_feature_name}_nwise_test_{split_name}.csv"

    if not overwrite and fullpath.exists():
        existed_log = f"{fullpath} existed, skipping statistics step."
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, existed_log)
    else:
        ttest = popa.compare_groupings(feature_results, popa.nwise_comparison_relational)
        ttest.to_csv(fullpath, sep='\t')
        ttest_n = len(ttest)
        dl.write_to_manifest_new_summary_file(dl.STATISTICS_TYPE, fullpath, f"{sanitized_feature_name}:{split_name}", f"{ttest_n}")


def analysis_precompute(res, collapse=True):
    self_labels, other_labels = sumr.fill_labels(res)
    self_features, other_features, general_features = sumr.fill_features(res)
    ratings = sumr.fill_ratings(res)
    performance_results = get_sample_aggregates(res, "performance_length", collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
    ratings_results = dict()
    for rating in ratings:
        ratings_results[rating] = get_sample_aggregates(res, rating, collapse=collapse, self_labels=self_labels, other_labels=other_labels, self_features=self_features, other_features=other_features, general_features=general_features)
    return collapse, self_labels, other_labels, self_features, other_features, general_features, ratings, performance_results, ratings_results

def run_statistics(overwrite=True, groupings=None, test_relational_features=True, test_self_features=True, collapse=True, tasks=None):
    speakers, samples = dfs.get_speakers_samples_full_dataframe(tasks)
    summaries = sumr.load_summaries(npyr.npy_list(), tasks=tasks)

    dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Running statistics (Speakers: {len(speakers)}, Samples: {len(samples)}, Summaries: {len(list(summaries.keys()))})", l=0)

    task_name = " ".join(sorted(nt.task_names_no_prefix(tasks)))
    for grouping in groupings:
        grouping_name = " & ".join(grouping)
        grouping_name = f"{grouping_name} ({task_name})"
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Started statistics for grouping: \'{grouping_name}\'")
        speaker_group = dfr.get_speaker_groups(speakers, grouping)
        group_count = len(speaker_group)
        dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Groups in \'{grouping_name}\': {group_count}")
        sample_groupings = dfr.find_sample_groupings(speaker_group, samples)
        sumr.replace_labels(summaries, "segment count", "segment_count")
        res = sumr.apply_sample_groupings_to_summaries(summaries, sample_groupings)

        collapse, self_labels, other_labels, self_features, other_features, general_features, ratings, performance_results, ratings_results = analysis_precompute(res, collapse=collapse)

        test_tasks = []
        if group_count == 2:
            if test_self_features: test_tasks.append((pairwise_test_self_features, SELF_FEATURES_LIST))
            if test_relational_features: test_tasks.append((pairwise_test_relational_features, RELATIONAL_FEATURES_LIST))
        if group_count > 2:
            if test_self_features: test_tasks.append((speakerwise_test_self, SELF_FEATURES_LIST))
            if test_relational_features: test_tasks.append((speakerwise_test_relational, RELATIONAL_FEATURES_LIST))
        if sumr.get_average_samples_count(sample_groupings) > 5:
            if test_self_features: test_tasks.append((nwise_test_self, SELF_FEATURES_LIST))
            if test_relational_features: test_tasks.append((nwise_test_relational, RELATIONAL_FEATURES_LIST))

        
        for test_method, feature_set  in test_tasks:
            try:
                test(test_method = test_method,
                    feature_set = np.copy(feature_set),
                    res = res,
                    collapse = collapse,
                    self_labels = np.copy(self_labels),
                    other_labels = np.copy(other_labels),
                    self_features = np.copy(self_features),
                    other_features = np.copy(other_features),
                    general_features = np.copy(general_features),
                    ratings = np.copy(ratings),
                    performance_results = performance_results,
                    ratings_results = ratings_results,
                    grouping_name=grouping_name
                    )
            except Exception as e:
                dl.write_to_manifest_log(dl.STATISTICS_TYPE, f"Running statistics failed for test: \'{test_method.__name__}\'")
                dl.log_stack("Failure in running experiments")
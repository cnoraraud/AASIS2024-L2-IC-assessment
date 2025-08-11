import concurrent
from scipy import stats as sstat
import numpy as np
import numpy_wrapper as npw
import npz_reader as npzr
import filtering as filt
import analysis as ana
import data_logger as dl
import naming_tools as nt


def apply_method_to_feature(data, method, properties=None):
    return method(data, properties=properties)


def apply_method_to_feature_at_i(data, method, i, properties=None):
    return apply_method_to_feature(data[i, :], method, properties=properties)


def apply_method_to_all_features(data, method, properties=None):
    return apply_method_to_select(
        data, method, np.any(~np.isnan(data), axis=1), properties=properties
    )


def apply_method_to_select(data, method, select, properties=None):
    if (
        properties is not None
        and "destructive" in properties
        and properties["destructive"]
    ):
        new_data_list = [None] * select.shape[0]
        max_length = 0
        for i in range(select.shape[0]):
            if select[i]:
                new_data_list[i] = apply_method_to_feature_at_i(
                    data, method, i, properties=properties
                )
                max_length = max(max_length, new_data_list[i].shape[0])
        new_data = np.empty((len(new_data_list), max_length))
        for i in range(select.shape[0]):
            if not isinstance(new_data_list[i], type(None)):
                new_data[i, : new_data_list[i].shape[0]] = new_data_list[i]
        return new_data
    else:
        new_data = np.copy(data)
        for i in range(select.shape[0]):
            if select[i]:
                new_data[i] = apply_method_to_feature_at_i(
                    data, method, i, properties=properties
                )
        return new_data


def analyze_feature(data, method, i, properties=None):
    return method(data[i, :], l=i, properties=properties)


def analyze_all_features(data, method, properties):
    ignore_nans = False
    if "ignore_nans" in properties:
        ignore_nans = properties["ignore_nans"]
    filter = np.full(data.shape[0], True)
    if ignore_nans:
        filter = np.any(~np.isnan(data), axis=1)
    return analyze_select(data, method, filter, properties=properties)


def try_analyze(data, method, select, i, properties=None):
    if select[i]:
        try:
            return analyze_feature(data, method, i, properties=properties), i
        except Exception as e:
            dl.log_stack(f"try_analyze exception at {method.__name__}:{i}\n\t{e}")
    return None, i


def analyze_select(data, method, select, properties=None):
    parallel = False
    if "parallel" in properties:
        parallel = properties["parallel"]

    if not parallel:
        results = [None] * select.shape[0]
        for i in range(select.shape[0]):
            results[i], _ = try_analyze(data, method, select, i, properties=properties)
        return results
    else:
        results = [None] * select.shape[0]
        executor = concurrent.futures.ProcessPoolExecutor(8)
        futures = [
            executor.submit(try_analyze, data, method, select, i, properties=properties)
            for i in range(select.shape[0])
        ]
        concurrent.futures.wait(futures)
        for future in futures:
            result, i = future.result()
            results[i] = result
        return results
    return None


# Not as precise but doesn't struggle with edge cases.
# Assumes it's always someone's turn
def turn_taking_times_comparative(D, L, n=5000, use_density=False):
    L_filter = npzr.has(L, "text:text") & npzr.double_speaker_filter(L)
    D, L = npzr.do_label_select(D, L, L_filter)
    if np.size(L) < 2:
        return np.zeros(0), np.full(0, nt.SPEAKERNONE)
    silence_mask = np.sum(D, axis=0) == 0.0
    if use_density:
        D = ana.apply_method_to_all_features(D, filt.to_density, {})
    D = ana.apply_method_to_all_features(D, filt.to_01, {})
    D[:, silence_mask] = np.nan
    D = ana.apply_method_to_all_features(D, filt.interpolate_nans, {})
    D = ana.apply_method_to_all_features(D, filt.ma, {"n": n})
    S1_larger = (D[0, :] >= D[1, :]).astype(np.int_)
    diff = np.diff(S1_larger)
    times = np.where(diff != 0)[0]
    S1_starting = diff[times] > 0
    starting = np.full(times.shape, nt.SPEAKER2)
    starting[S1_starting] = nt.SPEAKER1
    return times, starting


def find_pearsonr(a, channel_i, properties):
    B = properties["B"]
    results = [None] * B.shape[0]
    for i in range(B.shape[0]):
        b = B[i, :]
        results[i] = sstat.pearsonr(a, b)[0]
    return results


def find_spearmanr(a, channel_i, properties):
    B = properties["B"]
    results = [None] * B.shape[0]
    for i in range(B.shape[0]):
        b = B[i, :]
        results[i] = sstat.spearmanr(a, b)[0]
    return results


def get_segments(a):
    diff = np.diff(np.pad(a, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return starts, ends


# Confusing because the main feature isn't for the self but the other
def get_segment_comparisons(a, channel_i, properties):
    B = properties["B"]
    B2 = properties["B2"]
    N = B.shape[0]
    T = a.shape[0]
    peaked = B + a
    overlap = filt.flatten(peaked, {"threshold": 2})
    gap = filt.flatten(peaked, {"threshold": 0, "reverse": True}) + a - overlap
    a2 = B2[channel_i, :]
    # Find Segments
    starts, ends = get_segments(a)

    delays = []
    overlaps = []
    differences = []
    times_from_start = []
    times_from_end = []
    times_relative = []
    flat_widths = ends - starts

    for _ in range(N):
        delays.append(list())
        overlaps.append(list())
        differences.append(list())
        times_from_start.append(list())
        times_from_end.append(list())
        times_relative.append(list())

    # Calculate per Feature:
    for n in range(N):
        b = B[n, :]
        b2 = B2[n, :]
        b_overlap = overlap[n, :]
        b_gap = gap[n, :]
        b_overlap_starts, b_overlap_ends = get_segments(b_overlap)
        b_gap_starts, b_gap_ends = get_segments(b_gap)
        b_segment_starts, b_segment_ends = get_segments(b)

        overlap_j = 0
        gap_j = 0
        segment_j = 0
        for i in range(len(starts)):
            segment_overlaps = []
            segment_delays = []
            segment_differences = []
            segment_times_from_start = []
            segment_times_from_end = []
            segment_times_relative = []

            start = starts[i]
            next_start = T
            if i < len(starts) - 1:
                next_start = starts[i + 1]
            end = ends[i]
            segment_width = end - start
            segment_mass = np.nanmean(a2[start:end])
            # Skip overlaps outside the segment (should never actually happen)
            while overlap_j < len(b_overlap_ends) and b_overlap_ends[overlap_j] < start:
                overlap_j += 1
            # Go over every overlap inside the segment
            while (
                overlap_j < len(b_overlap_starts)
                and b_overlap_starts[overlap_j] < next_start
            ):
                # Register
                segment_overlap = (
                    b_overlap_ends[overlap_j] - b_overlap_starts[overlap_j]
                )
                segment_time_from_start = b_overlap_starts[overlap_j] - start
                segment_time_from_end = end - b_overlap_starts[overlap_j]
                segment_end_time_from_start = b_overlap_ends[overlap_j] - start
                segment_overlap_centre = (
                    segment_time_from_start + segment_end_time_from_start
                ) / 2
                segment_time_relative = np.nan
                if segment_width > segment_overlap_centre:
                    segment_time_relative = segment_overlap_centre / segment_width
                segment_overlaps.append(segment_overlap)
                segment_times_from_start.append(segment_time_from_start)
                segment_times_from_end.append(segment_time_from_end)
                segment_times_relative.append(segment_time_relative)
                # Go to next overlap
                overlap_j += 1

            # Skip gaps that end before the segment starts (will happen)
            while gap_j < len(b_gap_ends) and b_gap_ends[gap_j] < end:
                gap_j += 1
            # Go over every gap inside the segment
            while (
                gap_j < len(b_gap_ends)
                and b_gap_ends[gap_j] < next_start
                and b_gap_ends[gap_j] < T - 1
            ):
                segment_delay = b_gap_ends[gap_j] - end
                segment_delays.append(segment_delay)
                gap_j += 1

            while (
                segment_j < len(b_segment_starts)
                and b_segment_starts[segment_j] < next_start
            ):
                b_segment_start = b_segment_starts[segment_j]
                b_segment_end = b_segment_ends[segment_j]
                b_segment_mass = np.nanmean(b2[b_segment_start:b_segment_end])
                if npw.valid(b_segment_mass) and npw.valid(segment_mass):
                    mass_difference = b_segment_mass - segment_mass
                    mass_difference_relative = mass_difference / segment_mass
                    segment_differences.append(mass_difference_relative)

                segment_j += 1

            overlaps[n].append(np.array(segment_overlaps))
            delays[n].append(np.array(segment_delays))
            differences[n].append(np.array(segment_differences))
            times_from_start[n].append(np.array(segment_times_from_start))
            times_from_end[n].append(np.array(segment_times_from_end))
            times_relative[n].append(np.array(segment_times_relative))

    return {
        "overlaps": overlaps,
        "delays": delays,
        "differences": differences,
        "times_relative": times_relative,
        "times_from_start": times_from_start,
        "times_from_end": times_from_end,
        "flat_widths": flat_widths,
    }


def get_segment_masses_and_widths(a, channel_i, properties):
    # Find Segments
    starts, ends = get_segments((~np.isnan(a)).astype(np.int32))

    # Analyze Segments
    masses = []
    widths = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]

        mass = np.nansum(np.abs(a[start:end]))
        width = end - start
        masses.append(mass)
        widths.append(width)

    return {"masses": np.array(masses), "widths": np.array(widths)}


def get_all_correlations(labels, data):
    pearsoncorrs = np.corrcoef(data)
    spearmancorrs = np.array(analyze_all_features(data, find_spearmanr, {"B": data}))

    res = dict()
    for i in range(labels.shape[0]):
        label = labels[i]
        pearsoncorrs_row = pearsoncorrs[i, :]
        spearmancorrs_row = spearmancorrs[i, :]
        corrs = {"pearson_corrs": pearsoncorrs_row, "spearman_corrs": spearmancorrs_row}
        res[label] = corrs
    return res


def get_all_segment_comparisons(channel_indexes, data, mass_data):
    onds = analyze_all_features(
        data, get_segment_comparisons, {"B": data, "B2": mass_data}
    )
    res = dict()
    for channel_i, ond in zip(channel_indexes, onds):
        res[channel_i] = ond
    return res


def get_all_segment_masses_and_widths(channel_indexes, data):
    smws = analyze_all_features(data, get_segment_masses_and_widths, {"B": data})
    res = dict()
    for channel_i, smw in zip(channel_indexes, smws):
        res[channel_i] = smw
    return res


analysis_order = ["segment_comparisons", "masses_and_widths", "correllations"]


def group_analyses(L, *analyses):
    groups = []
    for i in range(L.shape[0]):
        label = L[i]
        group = {"label": label}
        for arg_num, analysis in enumerate(analyses):
            if isinstance(analysis, type(None)):
                dl.log(f"Analysis {analysis_order[arg_num]} missing.")
                continue
            if not isinstance(analysis, type(None)) and label in analysis:
                label_analysis = analysis[label]
                if not isinstance(label_analysis, type(None)):
                    group.update(label_analysis)
        groups.append(group)
    return groups


def channel_map(L, sub_L):
    L_map = dict()
    for i, label in enumerate(L):
        j_search = np.argwhere(sub_L == label).flatten()
        if j_search.size > 0:
            j = j_search[0]
            L_map[i] = j
    return L_map


def summarize_analyses(
    L, analyses, Ls, q=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - summary_reader.py get_labels
    L_map_sc = channel_map(L, Ls["segment_comparisons"])
    # L_map_mnw = label_map(L, Ls["masses_and_widths"])
    L_map_corr = channel_map(L, Ls["correlations"])

    summaries = dict()
    for a in analyses:
        self_summary = dict()
        label = None if "label" not in a else a["label"]
        masses = None if "masses" not in a else a["masses"]
        widths = None if "widths" not in a else a["widths"]
        flat_widths = None if "flat_widths" not in a else a["flat_widths"]
        overlaps = None if "overlaps" not in a else a["overlaps"]
        delays = None if "delays" not in a else a["delays"]
        differences = None if "differences" not in a else a["differences"]
        times_relative = None if "times_relative" not in a else a["times_relative"]
        times_from_start = (
            None if "times_from_start" not in a else a["times_from_start"]
        )
        times_from_end = None if "times_from_end" not in a else a["times_from_end"]
        pearson_corrs = None if "pearson_corrs" not in a else a["pearson_corrs"]
        spearman_corrs = None if "spearman_corrs" not in a else a["spearman_corrs"]

        density = npw.div_datas(masses, widths)

        self_summary["label"] = label
        self_summary["valid"] = npw.valid(widths) or npw.valid(masses)
        self_summary["segment_count"] = npw.count(widths)

        self_summary["total_segment_mass"] = npw.sum_data(masses)
        self_summary["total_segment_width"] = npw.sum_data(widths)

        self_summary["mean_segment_density"] = npw.mean_data(density)
        self_summary["mean_segment_width"] = npw.mean_data(widths)

        self_summary["median_segment_density"] = npw.median_data(density)
        self_summary["median_segment_width"] = npw.median_data(widths)

        self_summary["density_quantiles"] = npw.quantiles(density, q)
        self_summary["width_quantiles"] = npw.quantiles(widths, q)

        self_summary["std_segment_density"] = npw.std_data(density)
        self_summary["std_segment_width"] = npw.std_data(widths)

        self_summary["var_segment_density"] = npw.var_data(density)
        self_summary["var_segment_width"] = npw.var_data(widths)

        other_summaries = dict()
        for i in range(L.shape[0]):
            other_summary = dict()
            other_label = L[i]

            j_sc = None if i not in L_map_sc else L_map_sc[i]
            j_corr = None if i not in L_map_corr else L_map_corr[i]

            other_overlaps = (
                None
                if not npw.valid(overlaps) or isinstance(j_sc, type(None))
                else overlaps[j_sc]
            )
            other_delays = (
                None
                if not npw.valid(delays) or isinstance(j_sc, type(None))
                else delays[j_sc]
            )
            other_differences = (
                None
                if not npw.valid(differences) or isinstance(j_sc, type(None))
                else differences[j_sc]
            )
            other_times_relative = (
                None
                if not npw.valid(times_relative) or isinstance(j_sc, type(None))
                else times_relative[j_sc]
            )
            other_times_from_start = (
                None
                if not npw.valid(times_from_start) or isinstance(j_sc, type(None))
                else times_from_start[j_sc]
            )
            other_times_from_end = (
                None
                if not npw.valid(times_from_end) or isinstance(j_sc, type(None))
                else times_from_end[j_sc]
            )
            pearson_corr = (
                None
                if not npw.valid(pearson_corrs) or isinstance(j_corr, type(None))
                else pearson_corrs[j_corr]
            )
            spearman_corr = (
                None
                if not npw.valid(spearman_corrs) or isinstance(j_corr, type(None))
                else spearman_corrs[j_corr]
            )

            overlap_counts = npw.count_lengths(other_overlaps)
            overlap_sums = npw.sum_lengths(other_overlaps)
            delay_counts = npw.count_lengths(other_delays)
            other_overlap_percentage = npw.div_datas(overlap_sums, flat_widths)
            other_overlaps_all = npw.group_arrays(other_overlaps)
            other_delays_all = npw.group_arrays(other_delays)
            other_differences_all = npw.group_arrays(other_differences)
            other_times_relative_all = npw.group_arrays(other_times_relative)
            other_times_from_start_all = npw.group_arrays(other_times_from_start)
            other_times_from_end_all = npw.group_arrays(other_times_from_end)

            other_summary["label"] = label
            other_summary["other_label"] = other_label

            other_summary["count_overlap"] = npw.count(other_overlaps_all)
            other_summary["count_delay"] = npw.count(other_delays_all)

            other_summary["total_overlap"] = npw.sum_data(other_overlaps_all)

            other_summary["mean_overlap"] = npw.mean_data(other_overlaps_all)
            other_summary["mean_delay"] = npw.mean_data(other_delays_all)
            other_summary["mean_segment_overlap_count"] = npw.mean_data(overlap_counts)
            other_summary["mean_segment_delay_count"] = npw.mean_data(delay_counts)
            other_summary["mean_segment_overlap_ratio"] = npw.mean_data(
                other_overlap_percentage
            )
            other_summary["mean_difference"] = npw.mean_data(other_differences_all)
            other_summary["mean_times_relative"] = npw.mean_data(
                other_times_relative_all
            )
            other_summary["mean_times_from_start"] = npw.mean_data(
                other_times_from_start_all
            )
            other_summary["mean_times_from_end"] = npw.mean_data(
                other_times_from_end_all
            )

            other_summary["overlap_quantiles"] = npw.quantiles(other_overlaps_all, q)
            other_summary["delay_quantiles"] = npw.quantiles(other_delays_all, q)
            other_summary["segment_overlap_count_quantiles"] = npw.quantiles(
                overlap_counts, q
            )
            other_summary["segment_delay_count_quantiles"] = npw.quantiles(
                delay_counts, q
            )
            other_summary["segment_overlap_ratio_quantiles"] = npw.quantiles(
                other_overlap_percentage, q
            )
            other_summary["difference_quantiles"] = npw.quantiles(
                other_differences_all, q
            )
            other_summary["times_relative_quantiles"] = npw.quantiles(
                other_times_relative_all, q
            )
            other_summary["times_from_start_quantiles"] = npw.quantiles(
                other_times_from_start_all, q
            )
            other_summary["times_from_end_quantiles"] = npw.quantiles(
                other_times_from_end_all, q
            )

            other_summary["median_overlap"] = npw.median_data(other_overlaps_all)
            other_summary["median_delay"] = npw.median_data(other_delays_all)
            other_summary["median_segment_overlap_count"] = npw.median_data(
                overlap_counts
            )
            other_summary["median_segment_delay_count"] = npw.median_data(delay_counts)
            other_summary["median_segment_overlap_ratio"] = npw.median_data(
                other_overlap_percentage
            )
            other_summary["median_difference"] = npw.median_data(other_differences_all)
            other_summary["median_times_relative"] = npw.median_data(
                other_times_relative_all
            )
            other_summary["median_times_from_start"] = npw.median_data(
                other_times_from_start_all
            )
            other_summary["median_times_from_end"] = npw.median_data(
                other_times_from_end_all
            )

            other_summary["std_overlap"] = npw.std_data(other_overlaps_all)
            other_summary["std_delay"] = npw.std_data(other_delays_all)
            other_summary["std_segment_overlap_count"] = npw.std_data(overlap_counts)
            other_summary["std_segment_delay_count"] = npw.std_data(delay_counts)
            other_summary["std_segment_overlap_ratio"] = npw.std_data(
                other_overlap_percentage
            )
            other_summary["std_difference"] = npw.std_data(other_differences_all)
            other_summary["std_times_relative"] = npw.std_data(other_times_relative_all)
            other_summary["std_times_from_start"] = npw.std_data(
                other_times_from_start_all
            )
            other_summary["std_times_from_end"] = npw.std_data(other_times_from_end_all)

            other_summary["pearson_corr"] = pearson_corr
            other_summary["spearman_corr"] = spearman_corr

            other_summary["valid"] = (
                npw.valid(other_overlaps)
                or npw.valid(other_delays)
                or npw.valid(pearson_corrs)
                or npw.valid(spearman_corrs)
                or npw.valid(other_differences_all)
                or npw.valid(other_times_from_end_all)
                or npw.valid(other_times_from_start_all)
                or npw.valid(other_times_relative_all)
            )

            other_summaries[other_label] = other_summary

        summary = {"self": self_summary, "others": other_summaries}
        summaries[label] = summary
    return summaries

import numpy as np
import concurrent
from scipy import stats as sstat
import npz_reader as npzr

def apply_method_to_feature(data, method, properties, i):
    return method(data[i,:], properties)
def apply_method_to_all_features(data, method, properties):
    return apply_method_to_select(data, method, properties, (data==data)[:,0])
def apply_method_to_select(data, method, properties, select):
    new_data = np.copy(data)
    for i in range(select.shape[0]):
        if select[i]:
            new_data[i] = apply_method_to_feature(data, method, properties, i)
    return new_data
def analyze_feature(data, method, properties, i):
    return method(data[i,:], properties)
def analyze_all_features(data, method, properties):
    return analyze_select(data, method, properties, ((data==data)[:,0]))
def try_analyze(data, method, properties, select, i):
    if select[i]:
        try:
            return analyze_feature(data, method, properties, i), i
        except Exception as e:
            print(f"Exception at {i}:\n{e}")
    return None, i
def analyze_select(data, method, properties, select):
    parallel = False
    if "parallel" in properties:
        parallel = properties["parallel"]
    if not parallel:
        results = [None] * select.shape[0]
        for i in range(select.shape[0]):
            results[i], _ = try_analyze(data, method, properties, select, i)
        return results
    else:
        results = [None] * select.shape[0]
        executor = concurrent.futures.ProcessPoolExecutor(8)
        futures = [executor.submit(try_analyze, data, method, properties, select, i) for i in range(select.shape[0])]
        concurrent.futures.wait(futures)
        for future in futures:
            result, i = future.result()
            results[i] = result
        return results
    return None

def find_pearsonr(a, properties):
    B = properties["B"]
    results = [None] * B.shape[0]
    for i in range(B.shape[0]):
        b = B[i,:]
        results[i] = sstat.pearsonr(a, b)[0]
    return results
def find_spearmanr(a, properties):
    B = properties["B"]
    results = [None] * B.shape[0]
    for i in range(B.shape[0]):
        b = B[i,:]
        results[i] = sstat.spearmanr(a, b)[0]
    return results
def get_segments(a):
    diff = np.diff(np.pad(a,(1,1),"constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] -1
    return starts, ends
    
def get_segment_overlaps_and_delays(a, properties):
    B = properties["B"]
    N = B.shape[0]
    T = a.shape[0]
    peaked = B + a
    overlap = npzr.flatten_data(peaked,2)
    gap = npzr.flatten_data(peaked,0,reverse=True) + a - overlap
    # Find Segments
    starts, ends = get_segments(a)

    delays = []
    overlaps = []
    
    for n in range(N):
        delays.append(list())
        overlaps.append(list())
    
    # Calculate per Feature:
    for n in range(N):
        b = B[n,:]
        b_overlap = overlap[n,:]
        b_gap = gap[n,:]
        b_overlap_starts, b_overlap_ends = get_segments(b_overlap)
        b_gap_starts, b_gap_ends = get_segments(b_gap)
        
        overlap_j = 0
        gap_j = 0
        for i in range(len(starts)):
            segment_overlaps = []
            segment_delays = []
            start = starts[i]
            next_start = T
            if i < len(starts) - 1:
                next_start = starts[i + 1]
            end = ends[i]
            #Skip overlaps outside the segment (should never actually happen)
            while overlap_j < len(b_overlap_ends) and b_overlap_ends[overlap_j] < start:
                overlap_j += 1
            #Go over every overlap inside the segment
            while overlap_j < len(b_overlap_starts) and b_overlap_starts[overlap_j] < next_start:
                # Register
                segment_overlap = b_overlap_ends[overlap_j] - b_overlap_starts[overlap_j]
                segment_overlaps.append(segment_overlap)
                # Go to next overlap
                overlap_j += 1

            #Skip gaps that end before the segment starts (will happen)
            while gap_j < len(b_gap_ends) and b_gap_ends[gap_j] < end:
                gap_j += 1
            #Go over every gap inside the segment
            while gap_j < len(b_gap_ends) and b_gap_ends[gap_j] < next_start and b_gap_ends[gap_j] < T - 1:
                segment_delay = b_gap_ends[gap_j] - end
                segment_delays.append(segment_delay)
                gap_j += 1

            overlaps[n].append(np.array(segment_overlaps))
            delays[n].append(np.array(segment_delays))
    return {"overlaps": overlaps, "delays": delays}
    
def get_segment_masses_and_widths(a, properties):
    B = None
    N = -1
    if "B" in properties:
        B = properties["B"]
        N = B.shape[0]
    # Find Segments
    starts, ends = get_segments(a)
    
    # Analyze Segments
    masses = []
    widths = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        next_start = None
        if i < len(starts) - 1:
            next_start = starts[i+1]
        else:
            next_start = a.shape[0]
            
        mass = np.sum(np.abs(a[start:end]))
        width = end - start
        masses.append(mass)
        widths.append(width)

    return {"masses": np.array(masses), "widths": np.array(widths)}

def get_all_corellations(labels, data):
    pearsoncorrs = np.corrcoef(data)
    spearmancorrs = np.array(analyze_all_features(data, find_spearmanr, {"B":data}))

    all_corrs = []
    for i in range(labels.shape[0]):
        pearsoncorrs_row = pearsoncorrs[i, :]
        spearmancorrs_row = spearmancorrs[i, :]
        corrs = {"pearson_corrs": pearsoncorrs_row, "spearman_corrs": spearmancorrs_row}
        all_corrs.append(corrs)
    return all_corrs

def get_all_segment_overlaps_and_delays(labels, data):
    return analyze_all_features(data, get_segment_overlaps_and_delays, {"B": data})

def get_all_segment_masses_and_widths(labels, data):
    return analyze_all_features(data, get_segment_masses_and_widths, {"B": data})

def group_analyses(L, *args):
    groups = []
    for i in range(L.shape[0]):
        group = {"label": L[i]}
        for arg in args:
            group.update(arg[i])
        groups.append(group)
    return groups

def count_lengths(list_of_a):
    counts = []
    for a in list_of_a:
        counts.append(count(a))
    return np.array(counts)

def sum_lengths(list_of_a):
    sums = []
    for a in list_of_a:
        sums.append(sum_data(a))
    return np.array(sums)

def valid(data):
    if data is None or data.shape[0] == 0:
        return False
    return True

def quantiles(data, q, method="linear"):
    if not valid(data):
        return None
    return np.nanquantile(data, q=q, method=method)

def group_arrays(arrays):
    if len(arrays) == 0:
        return np.empty()
    return np.concat(arrays, axis=0)

def count(data):
    if not valid(data):
        return 0
    return data.shape[0]

def sum_data(data):
    if not valid(data):
        return 0
    return np.sum(data)

def mean_data(data):
    if not valid(data):
        return np.nan
    return np.nanmean(data)

def median_data(data):
    if not valid(data):
        return np.nan
    return np.nanmedian(data)

def summarize_analyses(L, analyses, q=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    summaries = dict()
    for a in analyses:
        self_summary = dict()
        label = None if "label" not in a else a["label"]
        masses = None if "masses" not in a else a["masses"]
        widths = None if "widths" not in a else a["widths"]
        overlaps = None if "overlaps" not in a else a["overlaps"]
        delays = None if "delays" not in a else a["delays"]
        pearson_corrs = None if "pearson_corrs" not in a else a["pearson_corrs"]
        spearman_corrs = None if "spearman_corrs" not in a else a["spearman_corrs"]

        density = masses/widths
        
        self_summary["label"] = label
        self_summary["valid"] = len(widths) > 0 or len(masses) > 0
        self_summary["segment count"] = count(widths)
        self_summary["total_segment_mass"] = sum_data(masses)
        self_summary["total_segment_width"] = sum_data(widths)
        self_summary["mean_segment_density"] = mean_data(density)
        self_summary["mean_segment_width"] = mean_data(widths)
        self_summary["width_quantiles"] = quantiles(widths, q)

        other_summaries = dict()
        for i in range(L.shape[0]):
            other_summary = dict()
            other_label = L[i]
            other_overlaps = overlaps[i]
            other_delays = delays[i]

            overlap_counts = count_lengths(other_overlaps)
            overlap_sums = sum_lengths(other_overlaps)
            delay_counts = count_lengths(other_delays)
            other_overlap_percentage = widths / overlap_sums
            other_overlaps_all = group_arrays(other_overlaps)
            other_delays_all = group_arrays(other_delays)

            other_summary["label"] = label
            other_summary["valid"] = len(other_overlaps) > 0 or len(other_delays) > 0
            other_summary["other_label"] = other_label
            other_summary["count_overlap"] = count(other_overlaps_all)
            other_summary["total_overlap"] = sum_data(other_overlaps_all)
            other_summary["mean_overlap"] = mean_data(other_overlaps_all)
            other_summary["median_overlap"] = median_data(other_overlaps_all)
            other_summary["mean_delay"] = mean_data(other_delays_all)
            other_summary["median_delay"] = median_data(other_delays_all)
            other_summary["mean_segment_overlap_ratio"] = mean_data(other_overlap_percentage)
            other_summary["median_segment_overlap_ratio"] = median_data(other_overlap_percentage)
            other_summary["segment_overlap_count_quantiles"] = quantiles(overlap_counts, q)
            other_summary["segment_delay_count_quantiles"] = quantiles(delay_counts, q)
            other_summary["overlap_quantiles"] = quantiles(other_overlaps_all, q)
            other_summary["delay_quantiles"] = quantiles(other_delays_all, q)
            other_summary["median_segment_overlap_count"] = median_data(overlap_counts)
            other_summary["median_segment_delay_count"] = median_data(delay_counts)
            other_summary["pearson_corr"] = pearson_corrs[i]
            other_summary["spearman_corr"] = spearman_corrs[i]
            
            other_summaries[other_label] = other_summary

        summary = {"self": self_summary, "others": other_summaries}
        summaries[label] = summary
    return summaries
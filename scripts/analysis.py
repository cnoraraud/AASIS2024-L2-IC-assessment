import concurrent
from scipy import stats as sstat
import numpy as np
import numpy_wrapper as npw
import npz_reader as npzr
import filtering as filt
import analysis as ana
import data_logger as dl

def apply_method_to_feature(data, method, properties = {}):
    return method(data, properties = properties)

def apply_method_to_feature_at_i(data, method, i, properties = {}):
    return apply_method_to_feature(data[i,:], method, properties = properties)

def apply_method_to_all_features(data, method, properties = {}):
    return apply_method_to_select(data, method, np.any(~np.isnan(data), axis=1), properties=properties)

def apply_method_to_select(data, method, select, properties = {}):
    if "destructive" in properties and properties["destructive"]:
        new_data_list = [None] * select.shape[0]
        max_length = 0
        for i in range(select.shape[0]):
            if select[i]:
                new_data_list[i] = apply_method_to_feature_at_i(data, method, i, properties=properties)
                max_length = max(max_length, new_data_list[i].shape[0])
        new_data = np.empty((len(new_data_list), max_length))
        for i in range(select.shape[0]):
            if not isinstance(new_data_list[i], type(None)):
                new_data[i, :new_data_list[i].shape[0]] = new_data_list[i]
        return new_data
    else:
        new_data = np.copy(data)
        for i in range(select.shape[0]):
            if select[i]:
                new_data[i] = apply_method_to_feature_at_i(data, method, i, properties=properties)
        return new_data

def analyze_feature(data, method, i, properties={}):
    return method(data[i,:], properties=properties)

def analyze_all_features(data, method, properties):
    ignore_nans = False
    if "ignore_nans" in properties:
        ignore_nans = properties["ignore_nans"]
    filter = np.full(data.shape[0], True)
    if ignore_nans:
        filter = np.any(~np.isnan(data), axis=1)
    return analyze_select(data, method, filter, properties=properties)

def try_analyze(data, method, select, i, properties={}):
    if select[i]:
        try:
            return analyze_feature(data, method, i, properties=properties), i
        except Exception as e:
            dl.log(f"try_analyze exception at {method.__name__}:{i}\n\t{e}")
    return None, i

def analyze_select(data, method, select, properties={}):
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
        futures = [executor.submit(try_analyze, data, method, select, i, properties=properties) for i in range(select.shape[0])]
        concurrent.futures.wait(futures)
        for future in futures:
            result, i = future.result()
            results[i] = result
        return results
    return None

# Not as precise but doesn't struggle with edge cases.
# Assumes it's always someone's turn
def turn_taking_times_comparative(D, L, n=5000, use_density=False):
    L_filter = npzr.has(L,"text:text") & npzr.double_speaker_filter(L)
    D, L = npzr.do_label_select(D, L, L_filter)
    if np.size(L) < 2:
        return np.zeros(0), np.full(0, npw.SPEAKERNONE)
    silence_mask = np.sum(D, axis=0) == 0.0
    if use_density: D = ana.apply_method_to_all_features(D, filt.to_density, {})
    D = ana.apply_method_to_all_features(D, filt.to_01, {})
    D[:, silence_mask] = np.nan
    D = ana.apply_method_to_all_features(D, filt.interpolate_nans, {})
    D = ana.apply_method_to_all_features(D, filt.ma, {"n": n})
    S1_larger = (D[0,:] >= D[1,:]).astype(np.int_)
    diff = np.diff(S1_larger)
    times = np.where(diff != 0)[0]
    S1_starting = diff[times] > 0
    starting = np.full(times.shape, npw.SPEAKER2)
    starting[S1_starting] = npw.SPEAKER1
    return times, starting

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
    overlap = filt.flatten(peaked, {"threshold": 2})
    gap = filt.flatten(peaked, {"threshold": 0, "reverse": True}) + a - overlap
    # Find Segments
    starts, ends = get_segments(a)

    delays = []
    overlaps = []
    flat_widths = ends - starts
    
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
    return {"overlaps": overlaps, "delays": delays, "flat_widths": flat_widths}
    
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

def get_all_correlations(labels, data):
    pearsoncorrs = np.corrcoef(data)
    spearmancorrs = np.array(analyze_all_features(data, find_spearmanr, {"B":data}))

    res = dict()
    for i in range(labels.shape[0]):
        label = labels[i]
        pearsoncorrs_row = pearsoncorrs[i, :]
        spearmancorrs_row = spearmancorrs[i, :]
        corrs = {"pearson_corrs": pearsoncorrs_row, "spearman_corrs": spearmancorrs_row}
        res[label] = corrs
    return res

def get_all_segment_overlaps_and_delays(labels, data):
    onds = analyze_all_features(data, get_segment_overlaps_and_delays, {"B": data})
    res = dict()
    for l, ond in zip(labels, onds):
        res[l] = ond
    return res

def get_all_segment_masses_and_widths(labels, data):
    smws = analyze_all_features(data, get_segment_masses_and_widths, {"B": data})
    res = dict()
    for l, smw in zip(labels, smws):
        res[l] = smw
    return res


analysis_order = ["overlaps_and_delays","masses_and_widths","correllations"]
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


def label_map(L, sub_L):
    L_map = dict()
    for i, l in enumerate(L):
        j_search = np.argwhere(sub_L == l).flatten()
        if j_search.size > 0:
            j = j_search[0]
            L_map[i] = j
    return L_map

def summarize_analyses(L, analyses, Ls, q=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - summary_reader.py get_labels
    L_map_ond = label_map(L, Ls["overlaps_and_delays"])
    L_map_mnw = label_map(L, Ls["masses_and_widths"])
    L_map_corr = label_map(L, Ls["correlations"])
    
    summaries = dict()
    for a in analyses:
        self_summary = dict()
        label = None if "label" not in a else a["label"]
        masses = None if "masses" not in a else a["masses"]
        widths = None if "widths" not in a else a["widths"]
        flat_widths = None if "flat_widths" not in a else a["flat_widths"]
        overlaps = None if "overlaps" not in a else a["overlaps"]
        delays = None if "delays" not in a else a["delays"]
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

            j_ond = None if i not in L_map_ond else L_map_ond[i]
            j_corr = None if i not in L_map_corr else L_map_corr[i]

            other_overlaps = None if not npw.valid(overlaps) or isinstance(j_ond, type(None)) else overlaps[j_ond]
            other_delays = None if not npw.valid(delays) or isinstance(j_ond, type(None)) else delays[j_ond]
            pearson_corr = None if not npw.valid(pearson_corrs) or isinstance(j_corr, type(None)) else pearson_corrs[j_corr]
            spearman_corr = None if not npw.valid(spearman_corrs) or isinstance(j_corr, type(None)) else spearman_corrs[j_corr]

            overlap_counts = npw.count_lengths(other_overlaps)
            overlap_sums = npw.sum_lengths(other_overlaps)
            delay_counts = npw.count_lengths(other_delays)
            other_overlap_percentage = npw.div_datas(overlap_sums, flat_widths)
            other_overlaps_all = npw.group_arrays(other_overlaps)
            other_delays_all = npw.group_arrays(other_delays)

            other_summary["label"] = label
            other_summary["other_label"] = other_label
            
            other_summary["count_overlap"] = npw.count(other_overlaps_all)
            other_summary["count_delay"] = npw.count(other_delays_all)

            other_summary["total_overlap"] = npw.sum_data(other_overlaps_all)

            other_summary["mean_overlap"] = npw.mean_data(other_overlaps_all)
            other_summary["mean_delay"] = npw.mean_data(other_delays_all)
            other_summary["mean_segment_overlap_count"] = npw.mean_data(overlap_counts)
            other_summary["mean_segment_delay_count"] = npw.mean_data(delay_counts)
            other_summary["mean_segment_overlap_ratio"] = npw.mean_data(other_overlap_percentage)
            
            other_summary["overlap_quantiles"] = npw.quantiles(other_overlaps_all, q)
            other_summary["delay_quantiles"] = npw.quantiles(other_delays_all, q)
            other_summary["segment_overlap_count_quantiles"] = npw.quantiles(overlap_counts, q)
            other_summary["segment_delay_count_quantiles"] = npw.quantiles(delay_counts, q)
            other_summary["segment_overlap_ratio_quantiles"] = npw.quantiles(other_overlap_percentage, q)
            
            other_summary["median_overlap"] = npw.median_data(other_overlaps_all)
            other_summary["median_delay"] = npw.median_data(other_delays_all)
            other_summary["median_segment_overlap_count"] = npw.median_data(overlap_counts)
            other_summary["median_segment_delay_count"] = npw.median_data(delay_counts)
            other_summary["median_segment_overlap_ratio"] = npw.median_data(other_overlap_percentage)
            
            other_summary["pearson_corr"] = pearson_corr
            other_summary["spearman_corr"] = spearman_corr

            other_summary["valid"] = npw.valid(other_overlaps) or npw.valid(other_delays) or npw.valid(pearson_corrs) or npw.valid(spearman_corrs)
            
            other_summaries[other_label] = other_summary

        summary = {"self": self_summary, "others": other_summaries}
        summaries[label] = summary
    return summaries
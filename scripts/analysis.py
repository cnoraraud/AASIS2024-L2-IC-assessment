import concurrent
import numpy as np
import numpy_wrapper as npw
import npz_reader as npzr
import filtering as filt
import analysis as ana
from scipy import stats as sstat

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
    return analyze_select(data, method, np.any(~np.isnan(data), axis=1), properties=properties)

def try_analyze(data, method, select, i, properties={}):
    if select[i]:
        try:
            return analyze_feature(data, method, i, properties=properties), i
        except Exception as e:
            print(f"Exception at {i}:\n{e}")
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
        for j, arg in enumerate(args):
            if arg[i] is None:
                print(f"Analysis {j} missing for label {L[i]}")
                continue
            group.update(arg[i])          
        groups.append(group)
    return groups

def summarize_analyses(L, analyses, q=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - summary_reader.py get_labels
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
        self_summary["segment count"] = npw.count(widths)
        
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
            other_overlaps = overlaps[i]
            other_delays = delays[i]

            overlap_counts = npw.count_lengths(other_overlaps)
            overlap_sums = npw.sum_lengths(other_overlaps)
            delay_counts = npw.count_lengths(other_delays)
            other_overlap_percentage = npw.div_datas(overlap_sums, flat_widths)
            other_overlaps_all = npw.group_arrays(other_overlaps)
            other_delays_all = npw.group_arrays(other_delays)

            other_summary["label"] = label
            other_summary["valid"] = npw.valid(other_overlaps) or npw.valid(other_delays)
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
            
            other_summary["pearson_corr"] = pearson_corrs[i]
            other_summary["spearman_corr"] = spearman_corrs[i]
            
            other_summaries[other_label] = other_summary

        summary = {"self": self_summary, "others": other_summaries}
        summaries[label] = summary
    return summaries
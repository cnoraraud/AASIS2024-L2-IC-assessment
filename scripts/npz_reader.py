import sys
import re
import math
import copy
from collections import Counter
from collections import defaultdict
import numpy as np
import io_tools as iot
import naming_tools as nt
import numpy_wrapper as npw
import analysis as ana
import filtering as filt
import data_logger as dl
import eaf_reader as eafr


def full(labels, value=True):
    return np.full(labels.shape, value)

def nothas(labels, sub):
    return np.char.find(np.array(labels), sub) == -1

def has(labels, sub):
    return np.char.find(np.array(labels), sub) != -1

def resolve_atomic(labels, atomic):
    negation = atomic[0]
    sub = atomic[1]
    if negation == "nothas":
        return nothas(labels, sub)
    if negation == "has":
        return has(labels, sub)

def resolve_conjunction(labels, conjunction_group):
    select = resolve_atomic(labels, conjunction_group[0])
    for i in range(1,len(conjunction_group)):
        select = select & resolve_atomic(labels, conjunction_group[i])
    return select

def canonical_select(labels, conjunction_groups):
    select = resolve_conjunction(labels, conjunction_groups[0])
    for i in range(1, len(conjunction_groups)):
        select = select | resolve_conjunction(labels, conjunction_groups[i])
    return select

def replace_labels(labels, sub, replacement):
    if np.size(labels) == 0:
        return labels
    return np.char.replace(np.array(labels), sub, replacement)

def do_label_select(data, labels, label_select):
    return data[label_select, :], labels[label_select]

def reorder_data(data, labels):
    if data.ndim == 1 or labels.ndim == 0:
        return data, labels
    order = np.argsort(labels, stable='True')
    return data[order, :], labels[order]

def do_time_select(data, time_select):
    return data[:, time_select]

def append_labels(data, labels, new_data, new_labels):
    return np.concat([data, new_data]), np.concat([labels, new_labels])

def focus_on_label(data, labels, label):
    select_filter = has(labels, label)
    time_filter = np.any(np.expand_dims(select_filter, axis=1) & ((data == 1)), axis=0)
    new_data = do_time_select(data, time_filter)
    return do_label_select(new_data, labels, ~select_filter)

def pmean(d, p=1):
    # found out scipy has this after I wrote it
    n = d.shape[0]
    dm = d
    if p == 0:
        dm = np.prod(d, axis=0, keepdims=True) ** (1 / n)
    elif math.isinf(p) and p > 0:
        dm = np.max(d, axis=0, keepdims=True)
    elif math.isinf(p) and p < 0:
        dm = np.min(d, axis=0, keepdims=True)
    else:
        dm = (np.nansum(d**p, axis=0, keepdims=True)/n) ** (1/p)
    return dm, p

# FINDS NUMBERS
def find_speaker_numbers(name):
    speakers = []
    matches = re.findall(r"speaker\d{3}",name)
    for match in matches:
        speaker = match.replace("speaker","")
        speakers.append(speaker)
    return speakers

def mode(candidates):
    return Counter(candidates).most_common(1)[0][0]

def anonymize_speakers(labels):
    source_map = dict()
    new_labels = []
    for label in labels:
        sources = nt.find_sources(label)
        anon_sources = []
        for source in sources:
            anon_source = nt.get_anon_source(source)
            anon_sources.append(anon_source)
            if source not in source_map:
                source_map[source] = anon_source
            if source_map[source] != anon_source:
                dl.log("Duplicated sources given same anonymous title.")
                dl.log("Possible presumptions broken:\n")
                dl.log("\t- Only 2 speakers present.")
                dl.log("\t- One speaker always even numbered, the other always odd")
        new_source = nt.compact_sources(anon_sources, plural_nicks=True)
        tag = nt.find_tag(label)
        feature = nt.find_channel(label)
        extra = nt.find_extra(label)
        new_label = nt.create_label(new_source, tag, feature, extra)
        new_labels.append(new_label)
    return npw.string_array(new_labels)

def strip_label_slots(labels, slots=[0, 1]):
    if not isinstance(slots, list):
        slots = [slots]
    new_labels = []
    for l in labels:
        l_slots = l.split(" ")
        new_slots = []
        for i, l_slot in enumerate(l_slots):
            if i not in slots: new_slots.append(l_slot)
        new_label = "-".join(new_slots)
        new_labels.append(new_label)
    return npw.string_array(new_labels)

def extract_speakers_DLs(data, labels):
    labels = anonymize_speakers(labels)
    S1_filter = has(labels, nt.SPEAKER1)
    S2_filter = has(labels, nt.SPEAKER2)
    D1, L1 = do_label_select(data, labels, S1_filter)
    D2, L2 = do_label_select(data, labels, S2_filter)
    L1 = strip_label_slots(L1)
    L2 = strip_label_slots(L2)
    return (D1, L1), (D2, L2)

def fit_DLs(DLs):
    L_common = set()
    for DL in DLs:
        D, L = DL
        L_common = L_common | set(L.tolist())
    L_common = npw.string_array(sorted(list(L_common)))
    Ds = []
    for DL in DLs:
        D, L = DL
        L_incommon = np.isin(L_common, L)
        D_incommon_shape = (L_common.shape[0], D.shape[1])
        D_incommon = np.full(D_incommon_shape, np.nan)
        D_incommon[L_incommon, :] = D
        Ds.append(D_incommon)
    return Ds, L_common

def dual_speaker_DLs(data, labels):
    DL1, DL2 = extract_speakers_DLs(data, labels)
    Ds, L_common = fit_DLs([DL1, DL2])
    return Ds[0], Ds[1], L_common

def double_speaker_filter(labels):
    return ~(has(labels, nt.SPEAKER1) & has(labels, nt.SPEAKER2))

def DL_info(D, L):
    return f"{D.shape}"

def write_DL(name, D, L, write_to_manifest=True, overwrite=True):
    if not overwrite:
        npz_path = iot.npzs_path() / nt.file_swap(name, "npz", all=False)
        if npz_path.exists():
            existed_log = f"{npz_path} existed, skipping DL step."
            dl.log(existed_log)
            return npz_path, [existed_log]
    
    eaf_path = iot.eafs_path() / nt.file_swap(name, "eaf", all=False)
    npz_path = iot.npzs_path() / nt.file_swap(name, "npz", all=False)
    np.savez(npz_path, D=D, L=L)
    input_info = None
    output_info = None
    try:
        input_info = eafr.eaf_info(eaf_path)
        output_info = DL_info(D, L)
    except Exception as e:
        dl.log_stack("Failed to get DL info.")
    if write_to_manifest:
        return npz_path, [dl.write_to_manifest_new_file(dl.DATA_TYPE, eaf_path, npz_path, input_info=input_info, output_info=output_info)]
    else:
        return npz_path, [None]

def npz_list():
    npzs = []
    for name in iot.list_dir(iot.npzs_path(), "npz"):
        npzs.append(name)
    return npzs

def print_npz():
    for name in npz_list():
        dl.log(name)

def print_annotation_info():
    npzs = npz_list()
    sources_to_ms = defaultdict(list)
    for npz in npzs:
        if "task5" not in npz:
            continue
        name, D, L = read_sanitized_DL_from_name(npz)
        ms = D.shape[1]
        canidates = nt.find_best_candidate(nt.find_speakers(name))
        source = nt.compact_sources(nt.speakers_to_sources(canidates))
        sources_to_ms[source].append(ms)
        del D
        del L
    recording_count = len(sources_to_ms)
    speaker_count = len(sources_to_ms) * 2
    annotation_count = 0
    overlap_speakers = 0
    total_ms = 0
    overlap_ms = 0
    for src in sources_to_ms:
        mss = sources_to_ms[src]
        for ms in mss:
            total_ms += ms
        if len(mss) > 1:
            overlap_ms += min(mss)
            overlap_speakers += 2
        annotation_count += len(mss)
    dl.log("ANNOTATION STATISTICS:")
    dl.log(f"- Recordings: {recording_count}")
    dl.log(f"- Annotations: {annotation_count}")
    dl.log(f"- Minutes: {(total_ms/1000)/60:.2f}")

def read_DL_metadata_from_name(name):
    name = nt.file_swap(name, "npz")
    return iot.read_metadata_from_path(iot.npzs_path() / name)
        
def read_DL_from_name(name):
    name = nt.file_swap(name, "npz")
    return read_DL(iot.npzs_path() / name)

def read_sanitized_DL(path):
    name, D, L = read_DL(path)
    L_anon = anonymize_speakers(L)
    D_s, L_anon = focus_on_label(D, L_anon, "performance")
    return name, D_s, L_anon

def read_sanitized_DL_from_name(name):
    name = nt.file_swap(name, "npz")
    return read_sanitized_DL(iot.npzs_path() / name)

def read_DL(path):
    npz = np.load(path)
    D = npz['D']
    L = npz['L']
    name = path.stem
    return name, D, L

def data_head(table, n = None):
    return data_section(table, n)

def data_section(table, start = 0, n = None):
    if n is None:
        n = len(table) - start
    return dict(list(table.items())[start : start + n]) # dictionaries are ordered as of python 3.7

def read_DLs(table, key_whitelist=None):
    all_names = []
    all_Ds = []
    all_Ls = []
    for key in table:
        if key_whitelist is None or key in key_whitelist:
            row = table[key]
            name, D, L = read_DL_from_name(key)
            all_names.append(name)
            all_Ds.append(D)
            all_Ls.append(L)
    return all_names, all_Ds, all_Ls

def get_DLs(table = None, n = None, start:int = 0):
    if table is None:
        table = generate_npz_table()
    return read_DLs(table = data_section(table, start, n))

def generate_npz_table():
    npzs = npz_list()
    table = dict()
    for npz in npzs:
        table[npz] = dict({"name": None, "D": None, "L": None})
    return table

class NpzProvider:
    npzs = []
    table = dict()

    def __len__(self):
        return len(self.npzs)
    
    def __init__(self):
        self.npzs = npz_list()
        self.table = generate_npz_table()
    
    def reset(self):
        self.pointer = 0

    def generator(self, n = 1):
        i = 0
        while i < len(self):
            yield get_DLs(self.table, n = n, start = i)
            i += n
    
    def nth_key(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        return list(rows.keys())[n]
    
    def nth(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        return rows[self.nth_key(rows, n)]
    
    def report_rows(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        for key in rows:
            row = rows[key]
            dl.log(f"[{key}]")

def get_data_segments(D, L, segments: list):
    # segments are in format {event: int, start: int, end: int}
    data_segments = []

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        data_segment = copy.deepcopy(segment)
        labels = copy.deepcopy(L)
        data = copy.deepcopy(D[:,start:end])
        starting_speaker = segment["speaker"]
        ending_speaker = nt.get_speaker_other(starting_speaker)
        labels = replace_labels(labels, starting_speaker, "new_speaker")
        labels = replace_labels(labels, ending_speaker, "old_speaker")
        data, labels = reorder_data(data, labels)
        data_segment["labels"] = labels
        data_segment["data"] = data
        data_segments.append(data_segment)

    return data_segments

def get_entire_data_as_segment(npz_name, selection):
    name, D, L = read_sanitized_DL_from_name(npz_name)
    if not isinstance(selection, type(None)):
        L_filter = canonical_select(L, selection)
        D, L = do_label_select(D, L, L_filter)
    start = 0
    end = 0
    if np.size(D) > 0:
        end = D.shape[-1]
    labels = copy.deepcopy(L)
    data = copy.deepcopy(D)
    segment = {"start": start,
               "end": end,
               "labels": labels,
               "data": data}
    return [segment]

def get_turn_taking_data_segments(npz_name, selection, n = 5000, use_density=False):
    name, D, L = read_sanitized_DL_from_name(npz_name)
    time_frames, starting_speakers = ana.turn_taking_times_comparative(D, L, n=n, use_density=use_density)
    start = 0
    end = D.shape[1]
    segments = []
    for time_frame, speaker in zip(time_frames, starting_speakers):
        segment_event = time_frame
        segment_start = max(start, time_frame - n)
        segment_end = min(end, time_frame + n)
        segments.append({"event": segment_event,
                         "start": segment_start,
                         "end": segment_end,
                         "speaker": speaker})

    if not isinstance(selection, type(None)):
        L_filter = canonical_select(L, selection)
        D, L = do_label_select(D, L, L_filter)
    data_segments = get_data_segments(D, L, segments)
    del D
    del L
    return data_segments

def get_turn_taking_features(times, starting, t_max):
    speakers = sorted(np.unique(starting).tolist())
    starting_list = starting.tolist()
    times_list = times.tolist()

    # Assume the first person to speak is the last person to speak next
    speaker_counter = copy.deepcopy(speakers)
    for starting_speaker in starting:
        if len(speaker_counter) == 1:
            break
        speaker_counter.remove(starting_speaker)
    assumed_first_speaker = speaker_counter[0]

    starting_list.insert(0,assumed_first_speaker)
    times_list.append(t_max)
    prev_time = 0
    D = np.zeros((len(speakers), t_max))
    for starting_speaker, t_frame in zip(starting_list, times_list):
        starting_index = prev_time
        end_index = t_frame
        speaker_index = speakers.index(starting_speaker)
        D[speaker_index, starting_index:end_index] = 1
        prev_time = t_frame

    labels = []
    for speaker in speakers:
        labels.append(nt.create_label(speaker, nt.EXTRACTION_TAG, f"{nt.AA_TAG}:turn"))
    L = npw.string_array(labels)
    
    return D, L

def add_turntaking(D, L):
    times, starting = ana.turn_taking_times_comparative(D, L, n=10000)
    tt_D, tt_L = get_turn_taking_features(times, starting, t_max=D.shape[1])
    D_concat = np.concat([D, tt_D], axis=0)
    L_concat = np.concat([L, tt_L], axis=0)
    return D_concat, L_concat

def get_all_segment_labels(segments):
    labels = set()
    for segment in segments:
        labels.update(set(segment["labels"].tolist()))
    return sorted(list(labels))

def get_max_segment_lengths(segments):
    max_dist_to_start = -1
    max_dist_to_end = -1
    max_dist_start_to_end = -1
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        max_dist_start_to_end = max(max_dist_start_to_end, abs(end - start))
        if "event" in segment:
            event = segment["event"]
            max_dist_to_start = max(max_dist_to_start, abs(event - start))
            max_dist_to_end = max(max_dist_to_end, abs(end - event))
    distance = -1
    max_dist_stitched = max_dist_to_start + max_dist_to_end
    if max_dist_start_to_end >= max_dist_stitched:
        anchor = max_dist_start_to_end/2
        distance = max_dist_start_to_end
    else:
        anchor = max_dist_to_start
        distance = max_dist_stitched
    
    # typically the anchor is the midpoint
    return distance, anchor

def combined_segment_matrix_with_anchor(segments):
    # Combined Matrix Initialization
    all_labels = npw.string_array(get_all_segment_labels(segments))
    max_segment_length, anchor = get_max_segment_lengths(segments)
    segment_count = len(segments)
    max_label_count = all_labels.shape[0]
    shape = (max_label_count, max_segment_length) 
    total_shape = shape + tuple([segment_count])
    total_matrix = np.full(total_shape, np.nan)

    # Combine based on anchor
    for i, segment in enumerate(segments):
        if np.size(segment["data"]) == 0:
            continue
        source_indecies = np.isin(segment["labels"], all_labels)
        target_indecies = np.isin(all_labels, segment["labels"])
        start = segment["start"]
        end = segment["end"]
        event = segment["event"]
        target_start = int(anchor - (event - start))
        target_end = int(anchor + (end - event))
        my_matrix = np.full(shape, np.nan)
        my_matrix[target_indecies, target_start:target_end] = segment["data"][source_indecies,:]
        total_matrix[:,:,i] = my_matrix
    
    return all_labels, total_matrix, anchor

def combined_segment_matrix_with_interpolation(segments):
    # Combined Matrix Initialization
    all_labels = npw.string_array(get_all_segment_labels(segments))
    max_segment_length, _ = get_max_segment_lengths(segments)
    segment_count = len(segments)
    max_label_count = all_labels.shape[0]
    shape = (max_label_count, max_segment_length) 
    total_shape = shape + tuple([segment_count])
    total_matrix = np.full(total_shape, np.nan)

    #Combine with interpolation
    for i, segment in enumerate(segments):
        if np.size(segment["data"]) == 0:
            continue

        source_indecies = np.isin(segment["labels"], all_labels)
        target_indecies = np.isin(all_labels, segment["labels"])

        my_matrix = np.full(shape, np.nan)
        interpolated_data = ana.apply_method_to_all_features(segment["data"], filt.fit_to_size, {"t": max_segment_length, "destructive": True})
        
        my_matrix[target_indecies, :] = interpolated_data[source_indecies, :]
        total_matrix[:,:,i] = my_matrix

    return all_labels, total_matrix

def running_combined_segment_matrix(segments):
    all_labels = npw.string_array(get_all_segment_labels(segments))
    max_segment_length, mid_point = get_max_segment_lengths(segments)
    segment_count = len(segments)
    max_label_count = all_labels.shape[0]
    shape = (max_label_count, max_segment_length)
    matrix = np.full(shape, 0.0)
    support_matrix = np.full(shape, 0.0)
    for i, segment in enumerate(segments):
        source_indecies = np.isin(segment["labels"], all_labels)
        target_indecies = np.isin(all_labels, segment["labels"])
        start = segment["start"]
        end = segment["end"]
        event = segment["event"]
        target_start = int(mid_point - (event - start))
        target_end = int(mid_point + (end - event))
        my_matrix = np.full(shape, np.nan)
        my_matrix[target_indecies, target_start:target_end] = segment["data"][source_indecies,:]
        
        nan_mask = ~np.isnan(my_matrix)
        matrix[nan_mask] += my_matrix[nan_mask]
        support_matrix[nan_mask] += 1
    return all_labels, matrix, support_matrix, mid_point

def get_all_channels():
    channels = set()
    for npz in npz_list():
        for label in read_DL_from_name(npz)[2].tolist():
            channels.add(nt.find_channel(label))
    return sorted(list(channels))

MAXDISCRETE = 50
def identify_data_type(D, L, th=MAXDISCRETE):
    discrete_channels = np.full(L.shape, True)
    binary_channels = np.full(L.shape, True)
    for i in range(D.shape[0]):
        label = L[i]
        is_discrete = True
        is_binary = True
        if "energy" in label or "ff:" in label or "cc:" in label:
            is_discrete = False
            is_binary = False
        elif "text:" in label:
            is_binary = False
        elif len(np.unique(D[i,:])) > th:
            is_discrete = False
            is_binary = False
        discrete_channels[i] = is_discrete
        binary_channels[i] = is_binary
    return discrete_channels, binary_channels

def get_D_smooth(D, k=3, n=3000):
    D_smooth = D
    for i in range(k):
        D_smooth = ana.apply_method_to_all_features(D_smooth, filt.ma, {"n": n})
    return D_smooth

def transform_to_continuous(D, discrete_channels, binary_channels, D_smooth=None):
    D_continuous = np.copy(D)
    if isinstance(D_smooth, type(None)):
        D_smooth = get_D_smooth(D_continuous[discrete_channels])
    else:
        D_smooth = D_smooth[discrete_channels]
    D_continuous[discrete_channels] = D_smooth
    return D_continuous

def transform_to_binary(D, discrete_channels, binary_channels, th=1, D_smooth=None):
    D_binary = np.copy(D)
    D_binary[discrete_channels] = filt.flatten(D_binary[discrete_channels], {})
    if isinstance(D_smooth, type(None)):
        D_smooth = get_D_smooth(D_binary[~discrete_channels])
    else:
        D_smooth = D_smooth[~discrete_channels]
    D_binary[~discrete_channels] = D_smooth
    D_binary[~discrete_channels] = ana.apply_method_to_all_features(D_binary[~discrete_channels], filt.flatten, {"threshold": th, "do_std": True, "do_centre": True})
    return D_binary

def transform_to_quantized_form(D, discrete_channels, binary_channels, n=MAXDISCRETE, D_smooth=None):
    D_quantized = np.copy(D)
    if isinstance(D_smooth, type(None)):
        D_smooth = get_D_smooth(D_quantized)
    D_quantized = D_smooth
    for i in range(D.shape[0]):
        ps = np.concatenate([[-np.inf], np.nanquantile(D_quantized[i, :], q=np.arange(n-1)/(n-2)), [np.inf]])
        for j in range(1, len(ps)):
            floor = ps[j-1]
            ceil = ps[j]
            bin_mask = (D_quantized[i, :] > floor + np.finfo(np.float64).eps) & (D_quantized[i, :] <= ceil)
            unique_values = len(np.unique(D_quantized[i, bin_mask]))
            quantized_value = np.nanmedian(D_quantized[i, bin_mask])
            if unique_values > 0:
                D_quantized[i, bin_mask] = quantized_value
    return D_quantized

def transform_to_thresholded(D, discrete_channels, binary_channels, th=1, D_smooth=None):
    D_thresholded = np.copy(D)
    if isinstance(D_smooth, type(None)):
        D_smooth = get_D_smooth(D_thresholded[~discrete_channels])
    else:
        D_smooth = D_smooth[~discrete_channels]
    D_thresholded[~discrete_channels] = D_smooth
    D_thresholded[~discrete_channels] = ana.apply_method_to_all_features(D_thresholded[~discrete_channels], filt.flatten, {"threshold": th, "do_std": True, "do_centre": True, "side": "sym"})
    threshold_mask = D_thresholded > 0
    D_thresholded[~threshold_mask] = np.nan
    D_thresholded[threshold_mask] = D[threshold_mask]
    return D_thresholded

def cut_badrows(D, L):
    nan_filt = np.any(~np.isnan(D) & ~np.isinf(D) & ~np.isclose(D,0), axis=1)
    return D[nan_filt, :], L[nan_filt]

def find_duplicate_indexes(L):
    unique_labels = []
    duplicate_groups = []
    for l in sorted(list(set(L.tolist()))):
        indexes = np.where(L == l)[0]
        if len(indexes) == 1:
            unique_labels.append(indexes[0])
        else:
            duplicate_groups.append(indexes)
    return unique_labels, duplicate_groups

def clean_duplicates(D, L, unique_labels, duplicate_groups):
    Ds = [D[unique_labels,:]]
    Ls = [L[unique_labels]]
    for duplicate_labels in duplicate_groups:
        max_D = np.nanmax(D[duplicate_labels], axis=0)
        min_D = np.nanmax(D[duplicate_labels], axis=0)
        abs_max_D = max_D
        min_D_highher_max = np.where(np.abs(max_D) < np.abs(min_D))
        abs_max_D[min_D_highher_max] = min_D[min_D_highher_max]
        duplicate_D = np.expand_dims(abs_max_D, axis=0)
        label = L[duplicate_labels][0]
        Ds.append(duplicate_D)
        Ls.append(np.expand_dims(label, axis=0))
    return np.concat(Ds, axis=0), np.concat(Ls, axis=0)

def cleaning(D, L):
    starting = L.shape[0]
    D, L = cut_badrows(D, L)
    ending = L.shape[0]
    bad_rows_cut = starting - ending

    starting = L.shape[0]
    ul, dg = find_duplicate_indexes(L)
    D, L = clean_duplicates(D, L, ul, dg)
    ending = L.shape[0]
    duplicates_cut = starting - ending

    starting = np.sum(np.isnan(D))
    D = np.nan_to_num(D, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
    ending = np.sum(np.isnan(D))
    infs_cut = ending - starting

    return D, L, bad_rows_cut, duplicates_cut, infs_cut

def clean_DL(name, D, L, overwrite=True):
    start_shape = D.shape
    D, L, bad_rows_cut, duplicates_cut, infs_cut = cleaning(D, L)
    _, _ = write_DL(name, D, L, write_to_manifest=False)
    end_shape = D.shape
    npz_path = iot.npzs_path() / nt.file_swap(name, "npz", all=False)

    return [dl.write_to_manifest_file_change("data_matrix", npz_path, start_shape, f"{end_shape} ({bad_rows_cut}, {duplicates_cut}; {infs_cut})", "clean_DL")]
        
if __name__ == '__main__':
    globals()[sys.argv[1]]()
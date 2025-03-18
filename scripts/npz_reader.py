import sys
import re
import math
import copy
import numpy as np
import io_tools as iot
import naming_tools as nt
import numpy_wrapper as npw
import analysis as ana
import filtering as filt
import data_logger as dl
import eaf_reader as eafr

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
                print("Duplicated sources given same anonymous title.")
                print("Possible presumptions broken:\n")
                print("\t- Only 2 speakers present.")
                print("\t- One speaker always even numbered, the other always odd")
        new_source = "-".join(anon_sources)
        tag = nt.find_tag(label)
        feature = nt.find_feature(label)
        extra = nt.find_extra(label)
        new_label = nt.create_label(new_source, tag, feature, extra)
        new_labels.append(new_label)
    return npw.string_array(new_labels)


def double_speaker_filter(labels):
    return ~(has(labels, npw.SPEAKER1) & has(labels, npw.SPEAKER2))

def DL_info(D, L):
    return f"{D.shape}"

def write_DL(name, D, L):
    eaf_path = iot.eafs_path() / nt.file_swap(name, "eaf", all=False)
    npz_path = iot.npzs_path() / nt.file_swap(name, "npz", all=False)
    np.savez(npz_path, D=D, L=L)
    input_info = eafr.eaf_info(eaf_path)
    output_info = DL_info(D, L)
    return [dl.write_to_manifest_new_file("data_matrix", eaf_path, npz_path, input_info=input_info, output_info=output_info)]

def npz_list():
    npzs = []
    for name in iot.list_dir(iot.npzs_path(), "npz"):
        npzs.append(name)
    return npzs

def print_npz():
    for name in npz_list():
        print(name)

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
            print(f"[{key}]")

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
        ending_speaker = npw.get_speaker_other(starting_speaker)
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
        labels.append(nt.create_label(speaker, nt.EXTRACTION_TAG, "ic:turn"))
    L = npw.string_array(labels)
    
    return D, L

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

def get_all_features():
    features = set()
    for npz in npz_list():
        for label in read_DL_from_name(npz)[2].tolist():
            features.add(nt.find_feature(label))
    return sorted(list(features))
        
if __name__ == '__main__':
    globals()[sys.argv[1]]()
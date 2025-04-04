import copy
import random
import numpy as np
import numpy_wrapper as npw
import npz_reader as npzr
import npy_reader as npyr
from collections import Counter

class DRows:
    
    def __init__(self, init_dict=None):
        self.labels = []
        self.values = []
        if isinstance(init_dict, dict):
            self.add_dictionary(init_dict)

    def value_to_string_array(self, value):
        vals = []
        if isinstance(value, np.ndarray):
            value = self.describe_array(value)
        if not isinstance(value, list):
            value = [value]
        for val in value:
            clean_val = None
            if npw.is_string(val):
                clean_val = val
            elif npw.is_int(val):
                clean_val = f"{val}"
            elif npw.is_float(val):
                clean_val = f"{val:.2f}"
                if clean_val[-1] == "0":
                    clean_val = clean_val[:-1]
            else:
                print(f"Unknown type \"{type(val)}\" for {val}")
            vals.append(clean_val)
        return vals

    def append(self, label, value):
        self.labels.append(label)
        self.values.append(self.value_to_string_array(value))

    def add_dictionary(self, dict_obj):
        for key in dict_obj:
            self.append(key, dict_obj[key])
    
    def vals_to_string(self, vals, max_val_len):
        new_vals = []
        for val in vals:
            new_vals.append(val.rjust(max_val_len, " "))
        return "\t".join(new_vals)
        
    def text(self):
        max_len = 0
        max_val_len = 0
        for l in self.labels:
            max_len = max(max_len, len(l))
        for vals in self.values:
            for val in vals:
                max_val_len = max(max_val_len, len(val))

        rows = []
        for l, vals in zip(self.labels, self.values):
            row = f"{l.rjust(max_len, " ")}:\t{self.vals_to_string(vals, max_val_len)}"
            rows.append(row)
        
        return "\n".join(rows)
    
    def describe_array(self, array):
        min_val = np.nanmin(array)
        max_val = np.nanmax(array)
        med_val = np.nanmedian(array)
        return [min_val, max_val, med_val]
    
def print_in_blocks(word_list, n = 4):
    i = -1
    for word in word_list:
        i += 1
        if (i + 1) % n == 0 or i + 1 == len(word_list):
            print(word)
        else:
            print(word, end="\t")

def get_summary_info_full(summary):
    name = summary["name"]
    features = summary["general"]["features"]
    length = summary["general"]["length"]
    performance_length = summary["general"]["performance_length"]
    non_nan_values = 0
    non_nan_fields = 0
    fields = 0
    for label_key in summary["label_summaries"]:
        self = summary["label_summaries"][label_key]["self"]
        others = summary["label_summaries"][label_key]["others"]
        for key in self.keys():
            fields += 1
            c = npw.count(self[key])
            non_nan_values += c
            if (c > 0):
                non_nan_fields += 1
        for others_key in others.keys():
            other = others[others_key]
            for key in other.keys():
                fields += 1
                c = npw.count(other[key])
                non_nan_values += c
                if (c > 0):
                    non_nan_fields += 1
    return name, features, performance_length, length, non_nan_fields, fields, non_nan_values

def describe_summary(summary):
    name, features, performance_length, length, non_nan_fields, fields, non_nan_values = get_summary_info_full(summary)
    drows = DRows()
    drows.append("name", name)
    drows.append("features", features)
    drows.append("length", f"{performance_length} ({length})")
    drows.append("valid fields", f"{non_nan_fields} ({fields})")
    drows.append("valid values", non_nan_values)
    print(drows.text())

def describe_summaries(summaries):
    describe_info_of_summaries(summaries)
    print()
    describe_datas_in_summaries(summaries)

def describe_info_of_summaries(summaries):
    feature_counts = []
    performance_lengths = []
    non_nan_field_counts = []
    field_counts = []
    non_nan_value_counts = []

    for summary_key in summaries:
        summary = summaries[summary_key]
        name, features, performance_length, length, non_nan_fields, fields, non_nan_values = get_summary_info_full(summary["summary"])
        feature_counts.append(features)
        performance_lengths.append(performance_length)
        non_nan_field_counts.append(non_nan_fields)
        field_counts.append(fields)
        non_nan_value_counts.append(non_nan_values)
    
    feature_counts = np.array(feature_counts)
    performance_lengths = np.array(performance_lengths)
    non_nan_field_counts = np.array(non_nan_field_counts)
    field_counts = np.array(field_counts)
    non_nan_value_counts = np.array(non_nan_value_counts)

    print("STATISTICS")
    drows = DRows()
    drows.append("summary count", len(summaries))
    drows.append("feature count", feature_counts)
    drows.append("performance length", performance_lengths)
    drows.append("valid fields", non_nan_field_counts)
    drows.append("total fields", field_counts)
    drows.append("valid values", non_nan_value_counts)
    print(drows.text())

def describe_datas_in_summaries(summaries):
    all_labels = get_labels(summaries)
    L1 = all_labels["L1"]
    L2 = all_labels["L2"]
    L = all_labels["L"]
    all_feature_names = get_feature_names(summaries)
    F1 = sorted(list(all_feature_names["F1"]))
    F2 = sorted(list(all_feature_names["F2"]))
    F = sorted(list(all_feature_names["F"]))
    
    print("LABELS")
    print_in_blocks(L)
    print()
    print("INDIVIDUAL_FEATURES")
    print_in_blocks(F1)
    print()
    print("RELATIONAL_FEATURES")
    print_in_blocks(F2)

def load_summaries(names):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - get labels
    summaries = dict()
    for name in names:
        npy_name, summary = npyr.load_summary_from_name(name)
        summaries[npy_name.name] = {"npy": npy_name.name, "summary":summary}
    return summaries

def count_label_values(summary):
    individual_values = 0
    individual_values_valid = 0
    relational_values = 0
    relational_values_valid = 0
    valid_structure = True
    try:
        label_summaries = summary["label_summaries"]
        for label_key in label_summaries:
            label_summary = label_summaries[label_key]
            label_summary_self = label_summary["self"]
            label_summary_others = label_summary["others"]
            individual_values += 1
            if label_summary_self["valid"]:
                individual_values_valid += 1
            for other_label_key in label_summary_others:
                label_summary_other = label_summary_others[other_label_key]
                relational_values += 1
                if label_summary_other["valid"]:
                    relational_values_valid += 1
    except:
        name = "unknown_summary"
        if "name" in summary:
            name = summary["name"]
        print(f"Summary for {name} found to have invalid structure")
        valid_structure = False
        
    return individual_values_valid, individual_values, relational_values_valid, relational_values, valid_structure

GET_OPTIONAL = "?"
GET_INDEX = "!"
GET_PATH = "/"
GET_BRANCH = "*"
def get(obj, path, supress_escape=False, fallback=None):
    if not supress_escape and GET_PATH in path:
        path = path.split(GET_PATH)
    if isinstance(path, str):
        path = [path]
    current_obj = obj
    for i in range(len(path)):
        keyword = path[i]
        
        # Loop restarted inside recursion
        if isinstance(keyword, str) and len(keyword) > 0:
            if keyword[0] == GET_BRANCH:
                contains_keyword = keyword[1:]
                sub_gets = []
                for sub_keyword in current_obj:
                    if contains_keyword not in sub_keyword:
                        continue
                    sub_path = [sub_keyword] + path[i+1:]
                    sub_get = get(current_obj, sub_path, fallback)
                    sub_gets.append(sub_get)
                return sub_gets
            elif keyword[0] == GET_OPTIONAL:
                optional_keyword = keyword[1:]
                has_optional_get = get(current_obj, [optional_keyword] + path[i+1:], fallback)
                nothas_optional_get = get(current_obj, path[i+1:], fallback)
                return [has_optional_get, nothas_optional_get]
            elif keyword[0] == GET_INDEX:
                value = None
                if len(keyword) > 1:
                    value = int(keyword[1:])
                key_list = list(current_obj.keys())
                indexed_keyword = None
                if value is None:
                    indexed_keyword = random.choice(key_list)
                else:
                    value = max(0, min(value, len(key_list) - 1))
                    indexed_keyword = key_list[value]
                return get(current_obj, [indexed_keyword] + path[i+1:], fallback)
        
        # Loop continues
        if isinstance(current_obj, dict) and keyword in current_obj:
            current_obj = current_obj[keyword]
        else:
            return fallback
    return current_obj

def flatten_to_list(structure):
    flattened_list = []
    for el in structure:
        if isinstance(el, list) or isinstance(el, list):
            flattened_list.extend(flatten_to_list(el))
        else:
            flattened_list.append(el)
    return list(flattened_list)

def flatten_to_set(structure):
    flattened_set = set()
    for el in structure:
        if isinstance(el, list) or isinstance(el, set):
            flattened_set.update(flatten_to_set(el))
        else:
            flattened_set.add(el)
    return set(flattened_set)

def get_labels(summaries):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - analysis.py summarize_analyses
    # - load_summaries
    res_labels_self = get(summaries, f"?*/?{GET_BRANCH}/?summary/?label_summaries/?{GET_BRANCH}/self/label")
    res_labels_others = get(summaries, f"?*/?{GET_BRANCH}/?summary/?label_summaries/?{GET_BRANCH}/others/{GET_BRANCH}/other_label")
    labels_1 = flatten_to_set([res_labels_self])
    labels_2 =  flatten_to_set([res_labels_others])
    labels_all = flatten_to_set([res_labels_self, res_labels_others])
    labels_1.discard(None)
    labels_2.discard(None)
    labels_all.discard(None)
    return {"L1": sorted(list(labels_1)), "L2": sorted(list(labels_2)), "L": sorted(list(labels_all))}

def get_shapes(obj):
    shape_counter = Counter()
    for el in flatten_to_list(obj):
        if npw.valid(el):
            shape_counter[np.shape(np.array(el))] += 1
    return shape_counter
def get_feature_sizes(summaries, feature_name):
    res_feature_self = get(summaries, f"?*/?*/?summary/?label_summaries/?*/self/{feature_name}")
    res_feature_others = get(summaries, f"?*/?*/?summary/?label_summaries/?*/others/*/{feature_name}")
    feature_shapes = get_shapes([res_feature_self, res_feature_others])
    return feature_shapes
def get_keys(obj):
    key_counter = Counter()
    for el in flatten_to_list(obj):
        if isinstance(el, dict):
            for key in el.keys():
                key_counter[key] += 1
    return key_counter
def get_feature_names(summaries):
    res_labels_self = get(summaries, f"?*/?*/?summary/?label_summaries/?*/self")
    res_labels_others = get(summaries, f"?*/?*/?summary/?label_summaries/?*/others/*")
    feature_names_1 = get_keys([res_labels_self])
    feature_names_2 = get_keys([res_labels_others])
    feature_names_all = get_keys([res_labels_self, res_labels_others])
    return {"F1": feature_names_1, "F2": feature_names_2, "F": feature_names_all}

def string_replace(old_value, find, replace):
    if isinstance(old_value,np.str_):
        return np.str_(old_value.replace(find, replace))
    return old_value.replace(find, replace)

def replace_labels(obj, find, replace):
    discard = []
    add = []
    for key in obj:
        value = obj[key]
        if isinstance(value, dict):
            replace_labels(value, find, replace)
        elif npw.is_string(value) and find in value:
            obj[key] = string_replace(value, find, replace)
        
        if npw.is_string(key) and find in key:
            discard.append(key)
            new_key = string_replace(key, find, replace)
            add.append({new_key:obj[key]})

    for key in discard:
        del obj[key]
    for key_value_pair in add:
        obj.update(key_value_pair)

def apply_sample_groupings_to_summaries(summaries, sample_groupings):
    summary_groupings = dict()
    no_sessions_found = set()
    for grouping in sample_groupings:
        count_total = 0
        count_not_found = 0
        samples = sample_groupings[grouping]
        group_summaries = dict()

        # Copy summaries to group summary
        for sample_key in samples:
            sample = samples[sample_key]
            npy = get(sample, "npy")
            npz = get(sample, "npz")
            speaker = get(sample, "speaker")
            S = get(sample, "S")
            found = False
            for summary_key in summaries:
                summary_obj = summaries[summary_key]
                summary = summary_obj['summary']
                
                # Match npy
                if npy != summary_obj['npy']:
                    continue
                found = True
                
                # Add npy, general, labels to new summary
                if npy not in group_summaries:
                    group_summaries[npy] = {'npy': copy.deepcopy(npy), 'summary':{'general': copy.deepcopy(summary['general'])}}

                # Copy label_summaries from old summary
                label_summaries = summary['label_summaries']
                for label_summary_key in label_summaries:
                    label_summary = label_summaries[label_summary_key]
                    # Match S1/S2
                    self_label = get(label_summary, 'self/label')
                    if npw.is_speaker_related(S, self_label):
                        group_summaries[npy]['summary'][self_label] = copy.deepcopy(label_summary)

                label_list = summary['labels']
                new_label_list = []
                for label in label_list:
                    if npw.is_speaker_related(S, label):
                        new_label_list.append(label)
                group_summaries[npy]['summary']['labels'] = npw.string_array(new_label_list)
                
            count_total += 1
            if not found:
                if speaker is None:
                    speaker = sample_key
                no_sessions_found.add(speaker)
                count_not_found += 1
        
        print(f"For grouping \"{grouping}\", {count_not_found} out of {count_total} speakers didn't have any recorded sessions.")
        summary_groupings[grouping] = group_summaries
    print(f"The speakers without recorded sessions were:")
    for speaker in sorted(list(no_sessions_found)):
        print(f"\t{speaker}")
    return summary_groupings

def sanitize_sample(sample, sanitize_value, sample_size):
    if not npw.valid(sample) or np.shape(sample) != sample_size:
        return np.full(sample_size, sanitize_value)
    return sample

def get_first_non_nan(obj, path):
    res = get(obj, path)
    res = flatten_to_list(res)
    if None in res:
        res.remove(None)
    return res[0]

def get_samples(summary, feature_name, self_labels=None, other_labels=None, sanitize=True, sanitize_value=np.nan, sample_size=()):
    # Single value per matrix (general)
    if isinstance(self_labels, type(None)):
        sample = get_first_non_nan(summary, f"summary/?label_summaries/general/{feature_name}")
        if sanitize:
            sample = sanitize_sample(sample, sanitize_value, sample_size)
        return sample
    
    matrix_shape = [len(self_labels)]
    if not isinstance(other_labels, type(None)):
        matrix_shape = matrix_shape + [len(other_labels)]
    matrix_shape = tuple(matrix_shape) + sample_size

    matrix = np.empty(matrix_shape)

    for i, l1 in enumerate(self_labels):
        if not isinstance(other_labels, type(None)):
            # Multiple values per label (other)
            for j, l2 in enumerate(other_labels):
                sample = get_first_non_nan(summary, f"summary/?label_summaries/{l1}/others/{l2}/{feature_name}")
                if sanitize:
                    sample = sanitize_sample(sample, sanitize_value, sample_size)
                matrix[i, j] = sample
        else:
            # Single value per label (self)
            sample = get_first_non_nan(summary, f"summary/?label_summaries/{l1}/self/{feature_name}")
            if sanitize:
                sample = sanitize_sample(sample, sanitize_value, sample_size)
            matrix[i] = sample
    return matrix

def gather_session_samples(summaries, feature_name, self_labels=None, other_labels=None, sanitize=True, sanitize_value=np.nan, sample_size=()):
    session_samples = {}

    self_labels = npw.string_array(self_labels)
    if not isinstance(other_labels, type(None)): 
        other_labels = npw.string_array(other_labels)

    for summary_key in summaries:
        summary = summaries[summary_key]
        sample = get_samples(summary, feature_name,
                             self_labels=self_labels, other_labels=other_labels,
                             sanitize=sanitize, sanitize_value=sanitize_value, sample_size=sample_size)
        npy = summary["npy"]
        session_samples[npy] = {"npy": npy, "sample": sample, "self_labels": self_labels, "other_labels": other_labels}
    
    return session_samples

def seperate_speaker_from_session(speaker, sample, self_labels, other_labels):
    self_mask = npzr.has(self_labels, speaker)
    new_sample = sample[self_mask]

    new_self_labels = npzr.replace_labels(npw.string_array(self_labels), speaker, "own")[self_mask]
    if not isinstance(other_labels, type(None)):
        new_other_labels = npzr.replace_labels(other_labels, speaker, "own")
        new_other_labels = npzr.replace_labels(npw.string_array(new_other_labels), npw.get_speaker_other(speaker), "other")
    else:
        new_other_labels = other_labels
    
    return new_sample, new_self_labels, new_other_labels

# Implicitly cuts out shared left labels
def collapse_sessions_to_speakers(session_samples):
    speaker_samples = dict()

    for sample_key in session_samples:
        session_sample = session_samples[sample_key]
        npy = session_sample["npy"]
        sample = session_sample["sample"]
        self_labels = session_sample["self_labels"]
        other_labels = session_sample["other_labels"]
        for speaker in npw.get_speakers():
            new_npy = f"{npy}_{speaker}"
            new_sample, new_self_labels, new_other_labels = seperate_speaker_from_session(speaker, sample, self_labels, other_labels)
            speaker_samples[new_npy] = {"npy": new_npy, "sample": new_sample, "self_labels": new_self_labels, "other_labels": new_other_labels}
    return speaker_samples
        
# Feature is already chosen, all samples have been forced to be the same size
def map_samples_to_common_matrix(all_samples, sanitize_value=np.nan, sample_size=()):
    all_self_labels = set()
    all_other_labels = set()
    for sample_key in all_samples:
        sample = all_samples[sample_key]
        all_self_labels.update(sample["self_labels"])
        all_other_labels.update(sample["other_labels"])
    all_self_labels = npw.string_array(sorted(list(all_self_labels)))
    all_other_labels = npw.string_array(sorted(list(all_other_labels)))

    shape = (len(all_samples), all_self_labels.shape[0], all_other_labels.shape[0]) + sample_size
    mega_matrix = np.full(shape=shape, fill_value=sanitize_value)

    for i, sample_key in enumerate(all_samples):
        sample = all_samples[sample_key]
        self_labels = sample["self_labels"]
        other_labels = sample["other_labels"]
        self_mask = np.isin(all_self_labels, self_labels)   
        other_mask = np.isin(all_other_labels, other_labels)

        mega_matrix[i, self_mask, other_mask] = sample["sample"]
    
    return mega_matrix

def map_samples_to_common_matrix(all_samples, all_self_labels=None, all_others_labels=None, sanitize_value=np.nan, sample_size=()):
    # Finding Labels
    find_self_labels = isinstance(all_self_labels, type(None))
    find_others_labels = isinstance(all_others_labels, type(None))
    if find_self_labels:
        all_self_labels = set()
    if find_others_labels:
        all_others_labels = set()
    for sample_key in all_samples:
        sample = all_samples[sample_key]
        if find_self_labels:
            all_self_labels.update(sample["self_labels"])
        other_labels = ["#"]
        if not isinstance(sample["other_labels"], type(None)):
            other_labels = sample["other_labels"]
        if find_others_labels:
            all_others_labels.update(other_labels)
    if find_self_labels:
        all_self_labels = sorted(list(all_self_labels))
    if find_others_labels:
        all_others_labels = sorted(list(all_others_labels))
    all_self_labels = npw.string_array(all_self_labels)
    all_others_labels = npw.string_array(all_others_labels)
    
    # Finding Matrix
    sample_shape = (len(all_self_labels), len(all_others_labels)) + sample_size
    shape = tuple([len(all_samples)]) + sample_shape
    mega_matrix = np.full(shape=shape, fill_value=sanitize_value)
    print(mega_matrix.shape)
    
    for i, sample_key in enumerate(all_samples):
        sample = all_samples[sample_key]
        self_labels = sample["self_labels"]
        other_labels = ["#"]
        if not isinstance(sample["other_labels"], type(None)):
            other_labels = sample["other_labels"]
        
        self_indecies = np.array(np.where(np.isin(all_self_labels, self_labels))[0])
        other_indecies = np.array(np.where(np.isin(all_others_labels, other_labels))[0])
        sample_matrix = np.full(shape=sample_shape,fill_value=sanitize_value)
        
        if not isinstance(sample["other_labels"], type(None)):
            sample_matrix[np.ix_(self_indecies, other_indecies)] = sample["sample"]
        else:
            sample_matrix[np.ix_(self_indecies, other_indecies)] = np.expand_dims(sample["sample"], axis=1)
        
        mega_matrix[i] = sample_matrix
    
    return mega_matrix, all_self_labels, all_others_labels

#TODO: add support for per-session-features (no self_labels or other_labels)
def get_data_matrix(sample_group, feature_name, self_labels=None, other_labels=None, collapse=False, relational_feature=None):
    size_mode = get_feature_sizes(sample_group, feature_name).most_common(1)[0][0]
    if isinstance(self_labels, type(None)) or isinstance(self_labels, type(None)):
        all_labels = get_labels(sample_group)
        L1 = all_labels["L1"]
        L2 = all_labels["L2"]
    if isinstance(self_labels, type(None)):
        self_labels = L1
    if isinstance(self_labels, type(None)):
        other_labels = L2
    if isinstance(relational_feature, type(None)):
        all_feature_names = get_feature_names(sample_group)
        F1 = sorted(list(all_feature_names["F1"]))
        F2 = sorted(list(all_feature_names["F2"]))
        if feature_name in F1 and feature_name in F2:
            print("Feature could be relational or individual, assuming individual.")
            relational_feature = False
        elif feature_name in F1:
            relational_feature = False
        elif feature_name in F2:
            relational_feature = True
        else:
            print("Feature not found, stopping...")
            return None, None, None
    
    if relational_feature:
        other_labels = None
    
    session_samples = gather_session_samples(sample_group, feature_name, self_labels=self_labels, other_labels=other_labels, sample_size=size_mode)
    if collapse:
        speaker_samples = collapse_sessions_to_speakers(session_samples)
        M, ML1, ML2 = map_samples_to_common_matrix(speaker_samples, sample_size=size_mode)
    else:
        M, ML1, ML2 = map_samples_to_common_matrix(session_samples, sample_size=size_mode)
    return M, ML1, ML2
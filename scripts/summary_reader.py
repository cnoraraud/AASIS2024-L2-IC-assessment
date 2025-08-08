import copy
import random
import numpy as np
import numpy_wrapper as npw
import npz_reader as npzr
import npy_reader as npyr
import data_logger as dl
from collections import Counter
import naming_tools as nt

class DRows:
    
    def __init__(self, init_dict=None, skip_bad=False):
        self.labels = []
        self.values = []
        self.skip_bad = skip_bad
        if isinstance(init_dict, dict):
            self.add_dictionary(init_dict)

    def value_to_string_array(self, value):
        vals = []
        if isinstance(value, np.ndarray):
            value = self.describe_array(value)
        if not isinstance(value, list):
            value = [value]
        if self.skip_bad:
            try:
                if not self.check_good(value):
                    value = []
            except:
                pass
        for val in value:
            clean_val = None
            if npw.is_string(val):
                clean_val = val
            elif npw.is_int(val):
                clean_val = f"{val}"
            elif npw.is_float(val):
                clean_val = f"{val:.3f}"
                if clean_val[-1] == "0":
                    clean_val = clean_val[:-1]
            else:
                dl.log(f"Unknown type \"{type(val)}\" for {val}")
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
    
    def check_good(self, vals):
        vals = np.array(vals)
        return np.sum(np.isclose(np.nan_to_num(vals), 0)) < len(vals)

    def text(self, skip_bad=False):
        max_len = 0
        max_val_len = 0
        for l in self.labels:
            max_len = max(max_len, len(l))
        for vals in self.values:
            for val in vals:
                max_val_len = max(max_val_len, len(val))

        rows = []
        for l, vals in zip(self.labels, self.values):
            if len(vals) == 0:
                continue
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
            dl.log(word)
        else:
            dl.log(word, end="\t")

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
    dl.log(drows.text())

def describe_summaries(summaries):
    describe_info_of_summaries(summaries)
    dl.log()
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

    dl.log("STATISTICS")
    drows = DRows()
    drows.append("summary count", len(summaries))
    drows.append("feature count", feature_counts)
    drows.append("performance length", performance_lengths)
    drows.append("valid fields", non_nan_field_counts)
    drows.append("total fields", field_counts)
    drows.append("valid values", non_nan_value_counts)
    dl.log(drows.text())

def describe_datas_in_summaries(summaries):
    all_channels = get_channels(summaries)
    L1 = all_channels["L1"]
    L2 = all_channels["L2"]
    L = all_channels["L"]
    all_feature_names = get_feature_names(summaries)
    F1 = sorted(list(all_feature_names["F1"]))
    F2 = sorted(list(all_feature_names["F2"]))
    FG = sorted(list(all_feature_names["FG"]))
    F = sorted(list(all_feature_names["F"]))
    
    dl.log("LABELS")
    print_in_blocks(L)
    dl.log()
    dl.log("INDIVIDUAL_FEATURES")
    print_in_blocks(F1)
    dl.log()
    dl.log("RELATIONAL_FEATURES")
    print_in_blocks(F2)
    dl.log()
    dl.log("SESSION_FEATURES")
    print_in_blocks(FG)

def load_summaries(names=None, tasks=None):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - get labels
    if isinstance(names, type(None)):
        names = npyr.npy_list()
    summaries = dict()
    for name in names:
        if not isinstance(tasks, type(None)):
            found_task = False
            for task in nt.task_names_with_prefix(tasks):
                if task in name.lower():
                    found_task = True
                    break
            if not found_task:
                continue
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
        dl.log(f"Summary for {name} found to have invalid structure")
        valid_structure = False
        
    return individual_values_valid, individual_values, relational_values_valid, relational_values, valid_structure

def get_average_samples_count(sample_groupings):
    sample_counts = []
    for sample_grouping_key in sample_groupings:
        sample_counts.append(len(sample_groupings[sample_grouping_key]))
    return np.mean(sample_counts)

def measure_dict_tree_size(root):
    stack = [root]
    depth = [0]
    counter = 0
    max_depth = 0
    while len(stack) > 0:
        el = stack.pop()
        d = depth.pop()
        max_depth = max(d, max_depth)
        counter += 1
        if isinstance(el, dict):
            for key in list(el.keys()):
                stack.append(el[key])
                depth.append(d + 1)
    return counter, max_depth

G_OPT = "?"
G_IDX = "!"
G_PTH = "/"
G_BRC = "*"
G_OPTBRC = G_OPT+G_BRC
G_STANDARD_SUMMARY_ACCES = f"{G_OPTBRC}/{G_OPTBRC}/{G_OPT}summary"
def get(obj, path, supress_escape=False, fallback=None):
    if not supress_escape and G_PTH in path:
        path = path.split(G_PTH)
    if isinstance(path, str):
        path = [path]
    current_obj = obj
    for i in range(len(path)):
        keyword = path[i]
        
        # Loop restarted inside recursion
        if isinstance(keyword, str) and len(keyword) > 0:
            if keyword[0] == G_BRC and (isinstance(current_obj, list) or isinstance(current_obj, dict)):
                contains_keyword = keyword[1:]
                sub_gets = []
                for sub_keyword in current_obj:
                    if contains_keyword not in sub_keyword:
                        continue
                    sub_path = [sub_keyword] + path[i+1:]
                    sub_get = get(current_obj, sub_path, fallback)
                    sub_gets.append(sub_get)
                return sub_gets
            elif keyword[0] == G_OPT:
                optional_keyword = keyword[1:]
                has_optional_get = get(current_obj, [optional_keyword] + path[i+1:], fallback)
                nothas_optional_get = get(current_obj, path[i+1:], fallback)
                return [has_optional_get, nothas_optional_get]
            elif keyword[0] == G_IDX:
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
        elif isinstance(keyword, str) and len(keyword) == 0:
            return current_obj
        
        # Loop continues
        if isinstance(current_obj, dict) and keyword in current_obj:
            current_obj = current_obj[keyword]
        else:
            return fallback
    return current_obj

def flatten_to_list(structure, skip_bad=True):
    flattened_list = []
    for el in structure:
        if isinstance(el, list) or isinstance(el, list):
            flattened_list.extend(flatten_to_list(el))
        elif not skip_bad or not isinstance(el, type(None)):
            flattened_list.append(el)
    return list(flattened_list)

def flatten_to_set(structure, skip_bad=True):
    flattened_set = set()
    for el in structure:
        if isinstance(el, list) or isinstance(el, set):
            flattened_set.update(flatten_to_set(el))
        elif not skip_bad or not isinstance(el, type(None)):
            flattened_set.add(el)
    return set(flattened_set)

def get_channels(summaries):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - analysis.py summarize_analyses
    # - load_summaries
    res_labels_self = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/{G_OPT}label_summaries/{G_OPTBRC}/self/label")
    res_labels_others = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/{G_OPT}label_summaries/{G_OPTBRC}/others/{G_BRC}/other_label")
    labels_1 = flatten_to_set([res_labels_self])
    labels_2 =  flatten_to_set([res_labels_others])
    labels_all = flatten_to_set([res_labels_self, res_labels_others])
    labels_1.discard(None)
    labels_2.discard(None)
    labels_all.discard(None)
    return {"L1": sorted(list(labels_1)), "L2": sorted(list(labels_2)), "L": sorted(list(labels_all))}


def get_npys(samples):
    return sorted(list(flatten_to_set(get(samples, "?*/?*/npy"))))

def get_shapes(obj):
    shape_counter = Counter()
    for el in flatten_to_list(obj):
        if npw.valid(el):
            shape_counter[np.shape(np.array(el))] += 1
    return shape_counter
def get_feature_sizes(summaries, feature_name):
    res_feature_self = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/{G_OPT}label_summaries/{G_OPTBRC}/self/{feature_name}")
    res_feature_others = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/{G_OPT}label_summaries/{G_OPTBRC}/others/{G_OPT}/{feature_name}")
    res_feature_general = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/general/{feature_name}")
    feature_shapes = get_shapes([res_feature_self, res_feature_others, res_feature_general])
    return feature_shapes
def get_keys(obj):
    key_counter = Counter()
    for el in flatten_to_list(obj):
        if isinstance(el, dict):
            for key in el.keys():
                key_counter[key] += 1
    return key_counter
def get_feature_names(summaries):
    res_labels_proactive = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/{G_OPT}label_summaries/{G_OPTBRC}/self")
    res_labels_reactive = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/{G_OPT}label_summaries/{G_OPTBRC}/others/{G_BRC}")
    res_labels_general = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/general")
    feature_names_1 = get_keys([res_labels_proactive])
    feature_names_2 = get_keys([res_labels_reactive])
    feature_names_general = get_keys([res_labels_general])
    feature_names_all = get_keys([res_labels_proactive, res_labels_reactive, res_labels_general])
    return {"F1": feature_names_1, "F2": feature_names_2, "FG": feature_names_general, "F": feature_names_all}

def get_rating_names(summaries):
    res_labels_general = get(summaries, f"{G_STANDARD_SUMMARY_ACCES}/general")
    feature_names_general = get_keys([res_labels_general])
    not_ratings = []
    for feature in feature_names_general:
        if "rating" not in feature:
            not_ratings.append(feature)
    for feature in not_ratings:
        del feature_names_general[feature]
    return feature_names_general

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
            ratings = get(sample, "ratings")
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
                    group_summaries[npy] = {'npy': copy.deepcopy(npy), 'speaker': copy.deepcopy(speaker), 'S': copy.deepcopy(S), 'summary':{'general': copy.deepcopy(summary['general'])}}
                    for rating_key in ratings:
                        group_summaries[npy]["summary"]["general"][f"rating_{rating_key}"] = ratings[rating_key]

                # Copy label_summaries from old summary
                label_summaries = summary['label_summaries']
                for label_summary_key in label_summaries:
                    label_summary = label_summaries[label_summary_key]
                    # Match S1/S2
                    initiator_label = get(label_summary, 'self/label')
                    group_summaries[npy]['summary'][initiator_label] = copy.deepcopy(label_summary)

                label_list = summary['labels']
                new_label_list = []
                for label in label_list:
                    new_label_list.append(label)
                group_summaries[npy]['summary']['labels'] = npw.string_array(new_label_list)
                
            count_total += 1
            if not found:
                if speaker is None:
                    speaker = sample_key
                no_sessions_found.add(speaker)
                count_not_found += 1
        
        if count_not_found > 0:
            dl.log(f"For grouping \"{grouping}\", {count_not_found} out of {count_total} speakers didn't have any recorded sessions.")
        summary_groupings[grouping] = group_summaries
    if len(no_sessions_found) > 0:
        dl.log(f"The speakers without recorded sessions were:")
        for speaker in sorted(list(no_sessions_found)):
            dl.log(f"\t{speaker}")
    else:
        dl.log(f"No missing sessions.")

    return summary_groupings

def sanitize_sample(sample, sanitize_value, sample_size):
    if not npw.valid(sample) or np.shape(sample) != sample_size:
        return np.full(sample_size, sanitize_value)
    return sample

def get_first_non_nan(obj, path):
    res = get(obj, path)
    res = flatten_to_list(res)
    if len(res) > 0:
        return res[0]
    return None

def get_samples(summary, feature_name, initiator_labels=None, responder_labels=None, sanitize=True, sanitize_value=np.nan, sample_size=()):
    # Single value per matrix (general)
    if isinstance(initiator_labels, type(None)):
        sample = get_first_non_nan(summary, f"summary/{G_OPT}label_summaries/general/{feature_name}")
        if sanitize:
            sample = sanitize_sample(sample, sanitize_value, sample_size)
        return sample
    
    matrix_shape = [len(initiator_labels)]
    if not isinstance(responder_labels, type(None)):
        matrix_shape = matrix_shape + [len(responder_labels)]
    matrix_shape = tuple(matrix_shape) + sample_size

    # Dimension 0: initiator; Dimension 1: responder (if present)
    matrix = np.empty(matrix_shape)

    for i, l1 in enumerate(initiator_labels):
        if not isinstance(responder_labels, type(None)):
            # Multiple values per label (responder)
            for j, l2 in enumerate(responder_labels):
                sample = get_first_non_nan(summary, f"summary/{G_OPT}label_summaries/{l1}/others/{l2}/{feature_name}")
                if sanitize:
                    sample = sanitize_sample(sample, sanitize_value, sample_size)
                matrix[i, j] = sample
        else:
            # Single value per label (initiator)
            sample = get_first_non_nan(summary, f"summary/{G_OPT}label_summaries/{l1}/self/{feature_name}")
            if sanitize:
                sample = sanitize_sample(sample, sanitize_value, sample_size)
            matrix[i] = sample
    return matrix

def gather_session_samples(summaries, feature_name, initiator_labels=None, responder_labels=None, sanitize=True, sanitize_value=np.nan, sample_size=()):
    session_samples = {}

    if not isinstance(initiator_labels, type(None)):
        initiator_labels = npw.string_array(initiator_labels)
    if not isinstance(responder_labels, type(None)): 
        responder_labels = npw.string_array(responder_labels)

    for summary_key in summaries:
        summary = summaries[summary_key]
        sample = get_samples(summary, feature_name,
                             initiator_labels=initiator_labels, responder_labels=responder_labels,
                             sanitize=sanitize, sanitize_value=sanitize_value, sample_size=sample_size)
        npy = summary["npy"]
        S = summary["S"]
        speaker = summary["speaker"]
        session_samples[npy] = {"npy": npy, "S": S, "speaker": speaker, "sample": sample, nt.INIT_L_KEY: initiator_labels, nt.RESP_L_KEY: responder_labels}
    
    return session_samples

def seperate_speaker_from_session(speaker, sample, initiator_labels, responder_labels, collapse="none"):
    other_speaker = nt.get_speaker_other(speaker)

    if not isinstance(initiator_labels, type(None)):
        new_initiator_labels = npw.string_array(initiator_labels)
        new_initiator_labels = npzr.replace_labels(new_initiator_labels, speaker, nt.SPEAKER_SELF)
        new_initiator_labels = npzr.replace_labels(new_initiator_labels, other_speaker, nt.SPEAKER_OTHER)
    else:
        new_initiator_labels = initiator_labels
    
    if not isinstance(responder_labels, type(None)):
        new_responder_labels = npw.string_array(responder_labels)
        new_responder_labels = npzr.replace_labels(new_responder_labels, speaker, nt.SPEAKER_SELF)
        new_responder_labels = npzr.replace_labels(new_responder_labels, other_speaker, nt.SPEAKER_OTHER)
    else:
        new_responder_labels = responder_labels
    
    # NOT COLLAPSED:
    # IS = I initiated; RS = I responded; IO = You initiated; RO = You responded
    #       RS  RO
    #   IS  1   1
    #   IO  1   0
    # COLLAPSE INITIATOR:
    #       RS  RO
    #   IS  1   1
    #   IO  X   0
    # COLLAPSE RESPONDER:
    #       RS  RO
    #   IS  1   X
    #   IO  1   0


    if isinstance(collapse, str) and collapse != nt.COLLAPSE_NONE:
        if collapse == nt.COLLAPSE_INITIATOR_LABELS:
            collapse_mask = npzr.nothas(new_initiator_labels, nt.SPEAKER_OTHER)
            new_sample = sample[collapse_mask]
            new_initiator_labels = new_initiator_labels[collapse_mask]
        if collapse == nt.COLLAPSE_RESPONDER_LABELS:
            collapse_mask = npzr.nothas(new_responder_labels, nt.SPEAKER_OTHER)
            new_sample = sample[:, collapse_mask]
            new_responder_labels = new_responder_labels[collapse_mask]
    else:
        new_sample = sample

    return new_sample, new_initiator_labels, new_responder_labels

# Implicitly cuts out shared left labels
def sessions_to_speakers(session_samples, collapse=False):
    speaker_samples = dict()

    for sample_key in session_samples:
        session_sample = session_samples[sample_key]
        npy = session_sample["npy"]
        S = session_sample["S"]
        speaker = session_sample["speaker"]
        sample = session_sample["sample"]
        initiator_labels = session_sample[nt.INIT_L_KEY]
        responder_labels = session_sample[nt.RESP_L_KEY]
        sample_name = f"{npy}_{S}"
        new_sample, new_initiator_labels, new_responder_labels = seperate_speaker_from_session(S, sample, initiator_labels, responder_labels, collapse=collapse)
        speaker_samples[sample_name] = {"npy": sample_name, "sample": new_sample, nt.INIT_L_KEY: new_initiator_labels, nt.RESP_L_KEY: new_responder_labels, "speaker": speaker}
    return speaker_samples
        
# Feature is already chosen, all samples have been forced to be the same size
def map_samples_to_common_matrix(all_samples, all_initiator_labels=None, all_responder_labels=None, sanitize_value=np.nan, sample_size=()):
    # Finding Labels
    find_initiator_labels = isinstance(all_initiator_labels, type(None))
    find_responder_labels = isinstance(all_responder_labels, type(None))
    
    if find_initiator_labels:
        all_initiator_labels = set()
    if find_responder_labels:
        all_responder_labels = set()
    all_session_labels = list()
    for sample_key in all_samples:
        sample = all_samples[sample_key]
        npy = sample["npy"]
        all_session_labels.append(npy)
        
        if find_initiator_labels:
            initiator_labels = ["#"]
            if not isinstance(sample[nt.INIT_L_KEY], type(None)):
                initiator_labels = sample[nt.INIT_L_KEY]
            all_initiator_labels.update(initiator_labels)
    
        if find_responder_labels:
            responder_labels = ["#"]
            if not isinstance(sample[nt.RESP_L_KEY], type(None)):
                responder_labels = sample[nt.RESP_L_KEY]
            all_responder_labels.update(responder_labels)
    if find_initiator_labels:
        all_initiator_labels = sorted(list(all_initiator_labels))
    if find_responder_labels:
        all_responder_labels = sorted(list(all_responder_labels))
    
    all_initiator_labels = npw.string_array(all_initiator_labels)
    all_responder_labels = npw.string_array(all_responder_labels)
    all_session_labels = npw.string_array(all_session_labels)
    
    # Finding Matrix
    sample_shape = (len(all_initiator_labels), len(all_responder_labels)) + sample_size
    shape = tuple([len(all_samples)]) + sample_shape
    mega_matrix = np.full(shape=shape, fill_value=sanitize_value)
    
    for i, sample_key in enumerate(all_samples):
        sample = all_samples[sample_key]
        initiator_labels = ["#"]
        responder_labels = ["#"]
        if not isinstance(sample[nt.INIT_L_KEY], type(None)):
            initiator_labels = sample[nt.INIT_L_KEY]
        if not isinstance(sample[nt.RESP_L_KEY], type(None)):
            responder_labels = sample[nt.RESP_L_KEY]
        
        self_indecies = np.array([np.where(all_initiator_labels == label)[0][0] for label in initiator_labels])
        other_indecies = np.array([np.where(all_responder_labels == label)[0][0] for label in responder_labels])

        sample_matrix = np.full(shape=sample_shape,fill_value=sanitize_value)
        
        sample_value = sample["sample"]
        
        if isinstance(sample[nt.INIT_L_KEY], type(None)):
            sample_value = np.expand_dims(sample_value, axis=0)
        if isinstance(sample[nt.RESP_L_KEY], type(None)):
            sample_value = np.expand_dims(sample_value, axis=1)
        
        sample_matrix[np.ix_(self_indecies, other_indecies)] = sample_value
        
        mega_matrix[i] = sample_matrix
    
    return mega_matrix, all_session_labels, all_initiator_labels, all_responder_labels

def find_feature_type(self_features, others_features, general_features, feature_name):
    feature_type = None
    if sum([feature_name in self_features, feature_name in others_features, feature_name in general_features]) > 1:
            dl.log("Multiple options for feature.")
    if feature_name in self_features:
        feature_type = nt.PROACTIVE
    elif feature_name in others_features:
        feature_type = nt.REACTIVE
    elif feature_name in general_features:
        feature_type = nt.GENERAL
    return feature_type

def replace_if_none(origianl_value, replacement_value):
    if isinstance(origianl_value, type(None)):
        return replacement_value
    return origianl_value

def fill_labels(sample_group, initiator_labels=None, responder_labels=None):
    if isinstance(initiator_labels, type(None)) or isinstance(initiator_labels, type(None)):
        all_labels = get_channels(sample_group)
        L1 = all_labels["L1"]
        L2 = all_labels["L2"]
        initiator_labels = replace_if_none(initiator_labels, L1)
        responder_labels = replace_if_none(responder_labels, L2)
    return initiator_labels, responder_labels
            
def fill_features(sample_group, proactive_features=None, reactive_features=None, general_features=None):
    if isinstance(proactive_features, type(None)) or isinstance(reactive_features, type(None)) or isinstance(general_features, type(None)):
        all_feature_names = get_feature_names(sample_group)
        F1 = sorted(list(all_feature_names["F1"]))
        F2 = sorted(list(all_feature_names["F2"]))
        FG = sorted(list(all_feature_names["FG"]))
        proactive_features = replace_if_none(proactive_features, F1)
        reactive_features = replace_if_none(reactive_features, F2)
        general_features = replace_if_none(general_features, FG)
    return proactive_features, reactive_features, general_features

def fill_ratings(sample_group, ratings=None):
    if isinstance(ratings, type(None)):
        ratings = sorted(list(get_rating_names(sample_group)))
    return ratings

def find_sample_size_and_feature_type(sample_group, feature_name, proactive_features=None, reactive_features=None, general_features=None, sample_size=None, feature_type=None):
    proactive_features, reactive_features, general_features = fill_features(sample_group, proactive_features=proactive_features, reactive_features=reactive_features, general_features=general_features)
    if isinstance(sample_size, type(None)):
        sample_size = tuple([])
        all_sample_sizes = get_feature_sizes(sample_group, feature_name)
        if len(all_sample_sizes) > 0:
            sample_size = get_feature_sizes(sample_group, feature_name).most_common(1)[0][0]
    if isinstance(feature_type, type(None)):
        feature_type = find_feature_type(proactive_features, reactive_features, general_features, feature_name)
    return sample_size, feature_type

def get_data_matrix(sample_group, feature_name, initiator_labels=None, responder_labels=None, collapse=False, sample_size=None, feature_type=None, proactive_features=None, reactive_features=None, general_features=None):
    initiator_labels, responder_labels = fill_labels(sample_group, initiator_labels=initiator_labels, responder_labels=responder_labels)
    proactive_features, reactive_features, general_features = fill_features(sample_group, proactive_features=proactive_features, reactive_features=reactive_features, general_features=general_features)
    sample_size, feature_type = find_sample_size_and_feature_type(sample_group, feature_name, proactive_features=proactive_features, reactive_features=reactive_features, general_features=general_features)
    
    if feature_type == nt.PROACTIVE or feature_type == nt.GENERAL:
        responder_labels = None
    if feature_type == nt.GENERAL:
        initiator_labels = None
    
    if isinstance(feature_type, type(None)):
        return None, None, None, None
    
    samples = gather_session_samples(sample_group, feature_name, initiator_labels=initiator_labels, responder_labels=responder_labels, sample_size=sample_size)
    samples = sessions_to_speakers(samples, collapse=collapse)
    M, ML0, ML1, ML2 = map_samples_to_common_matrix(samples, sample_size=sample_size)
    return M, ML0, ML1, ML2
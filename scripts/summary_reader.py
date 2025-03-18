import copy
import random
import numpy as np
import numpy_wrapper as npw
import npz_reader as npzr
import npy_reader as npyr
import analysis as ana

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

def get(obj, path, supress_escape=False, search_in_children=False, fallback=None):
    if not supress_escape and "/" in path:
        path = path.split("/")
    if isinstance(path, str):
        path = [path]
    current_obj = obj
    for i in range(len(path)):
        keyword = path[i]
        if keyword == "*":
            sub_gets = []
            for sub_key in current_obj:
                sub_path = [sub_key] + path[i+1:]
                sub_get = get(current_obj, sub_path, search_in_children, fallback)
                sub_gets.append(sub_get)
            return sub_gets
        if isinstance(keyword, str) and len(keyword) > 0 and keyword[0] == "?":
            value = None
            if len(keyword) > 1:
                value = int(keyword[1:])
            key_list = list(current_obj.keys())
            key = None
            if value is None:
                key = random.choice(key_list)
            else:
                value = max(0, min(value, len(key_list) - 1))
                key = key_list[value]
            return get(current_obj, [key] + path[i+1:], search_in_children, fallback)

        elif keyword in current_obj:
            current_obj = current_obj[keyword]
        elif search_in_children:
            #TODO (low prio): Would be nice to search necessary stuff in children if can't find in yourself, looser pathing...
            raise NotImplementedError("search_in_children is not implemented")
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
    my_label = get(summaries, f"*/summary/*/self/label")
    other_labels = get(summaries, f"*/summary/*/others/*/other_label")
    labels_1 = flatten_to_set([my_label])
    labels_2 =  flatten_to_set([other_labels])
    labels_all = flatten_to_set([my_label, other_labels])
    labels_1.discard(None)
    labels_2.discard(None)
    labels_all.discard(None)
    return {"L1": sorted(list(labels_1)), "L2": sorted(list(labels_2)), "L": sorted(list(labels_all))}

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
    if not ana.valid(sample) or np.shape(sample) != sample_size:
        return np.full(sample_size, sanitize_value)
    return sample

def get_samples(summary, feature_name, self_labels=None, other_labels=None, sanitize=True, sanitize_value=np.nan, sample_size=()):
    # Single value per matrix (general)
    if isinstance(self_labels, type(None)):
        sample = get(summary, f"summary/general/{feature_name}")
        if sanitize:
            sample = sanitize_sample(sample, sanitize_value, sample_size)
        return sample
    
    matrix_shape = [self_labels.shape[0]]
    if not isinstance(other_labels, type(None)):
        matrix_shape = matrix_shape + [other_labels.shape[0]]
    matrix_shape = tuple(matrix_shape) + sample_size

    matrix = np.empty(matrix_shape)

    for i, l1 in enumerate(self_labels):
        if not isinstance(other_labels, type(None)):
            # Multiple values per label (other)
            for j, l2 in enumerate(other_labels):
                sample = get(summary, f"summary/label_summaries/{l1}/others/{l2}/{feature_name}")
                if sanitize:
                    sample = sanitize_sample(sample, sanitize_value, sample_size)
                matrix[i, j] = sample
        else:
            # Single value per label (self)
            sample = get(summary, f"summary/label_summaries/{l1}/self/{feature_name}")
            if sanitize:
                sample = sanitize_sample(sample, sanitize_value, sample_size)
            matrix[i] = sample
    return matrix

def gather_session_samples(summaries, feature_name, self_labels=None, other_labels=None, sanitize=True, sanitize_value=np.nan, sample_size=()):
    session_samples = {}

    for summary_key in summaries:
        npy = summary["npy"]
        summary = summaries[summary_key]
        sample = get_samples(summary, feature_name, self_labels=self_labels, other_labels=other_labels, sanitize=sanitize, sanitize_value=sanitize_value, sample_size=sample_size)
        session_samples[npy] = {"npy": npy, "sample": sample, "self_labels": self_labels, "other_labels": other_labels}

def seperate_speaker_from_session(speaker, sample, self_labels, other_labels):
    self_mask = npzr.has(self_labels, speaker)
    new_sample = sample[self_mask, :]
    new_self_labels = self_labels[self_mask]
    new_self_labels = npzr.replace_labels(new_self_labels, speaker, "own")
    speaker_other = npzr.get_speaker_other(speaker)
    new_other_labels = npzr.replace_labels(other_labels, speaker_other, "other")
    return new_sample, new_self_labels, new_other_labels

def collapse_sessions_to_speakers(session_samples):
    speaker_samples = {}

    for sample_key in session_samples:
        session_sample = session_samples[sample_key]
        npy = session_sample["npy"]
        sample = session_sample["sample"]
        self_labels = session_sample["self_labels"]
        other_labels = session_sample["other_labels"]
        for speaker in npzr.get_speakers():
            new_npy = f"{npy}_{speaker}"
            new_sample, new_self_labels, new_other_labels = seperate_speaker_from_session(speaker, sample, self_labels, other_labels)
            speaker_samples[new_npy] = {"npy": new_npy, "sample": new_sample, "self_labels": new_self_labels, "other_labels": new_other_labels}
    return speaker_samples
        
def map_samples_to_common_matrix(all_samples):
    all_self_labels = set()
    all_other_labels = set()
    for sample_key in all_samples:
        sample = all_samples[sample_key]
        all_self_labels.update(set(sample["self_labels"].tolist()))
        all_other_labels.update(set(sample["other_labels"].tolist()))
    all_self_labels = npw.string_array(sorted(list(all_self_labels)))
    all_other_labels = npw.string_array(sorted(list(all_other_labels)))

    shape = ()
    mega_matrix = np.full(shape=shape)

    for sample_key in all_samples:
        sample = all_samples[sample_key]
        self_labels = sample["self_labels"]
        other_labels = sample["other_labels"]
        self_mask = np.isin(all_self_labels, self_labels)   
        other_mask = np.isin(all_other_labels, other_labels)
    
    #TODO: The big one
        
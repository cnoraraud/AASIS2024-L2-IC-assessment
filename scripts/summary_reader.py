import npy_reader as npyr
import numpy as np
import random
import copy

def load_summaries(names):
    # volatile function, when you make changes, make sure you also make changes in reflective functions
    # - get labels
    summaries = dict()
    for name in names:
        npy_name, summary = npyr.load_summary_from_name(name)
        summaries[npy_name.name] = {"npy": npy_name.name, "summary":summary}
    return summaries

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
            #TODO
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
    label_collection = []
    my_label = get(summaries, f"*/summary/*/self/label")
    other_labels = get(summaries, f"*/summary/*/others/*/other_label")
    labels = flatten_to_set([my_label,other_labels])
    labels.discard(None)
    return labels

def string_replace(old_value, find, replace):
    if isinstance(old_value,np.str_):
        return np.str_(old_value.replace(find, replace))
    return old_value.replace(find, replace)

def is_string(value):
    return isinstance(value, str) or isinstance(value, np.str_)

def replace_labels(obj, find, replace):
    discard = []
    add = []
    for key in obj:
        value = obj[key]
        if isinstance(value, dict):
            replace_labels(value, find, replace)
        elif is_string(value) and find in value:
            obj[key] = string_replace(value, find, replace)
        
        if is_string(key) and find in key:
            discard.append(key)
            new_key = string_replace(key, find, replace)
            add.append({new_key:obj[key]})

    for key in discard:
        del obj[key]
    for key_value_pair in add:
        obj.update(key_value_pair)

def is_speaker_related(speaker, value):
    check_substring = value.split(" ")[0]
    if speaker in check_substring: return True
    if "[all]" in check_substring: return True
    return False

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
                    if is_speaker_related(S, self_label):
                        group_summaries[npy]['summary'][self_label] = copy.deepcopy(label_summary)

                label_list = summary['labels']
                new_label_list = []
                for label in label_list:
                    if is_speaker_related(S, label):
                        new_label_list.append(label)
                group_summaries[npy]['summary']['labels'] = np.asarray(new_label_list, dtype = '<U64')
                
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
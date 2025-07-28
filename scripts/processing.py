import sys
import math
import numpy as np
from collections import Counter
import data_reader as dr
import npz_reader as npzr
import eaf_reader as eafr
import wav_reader as wavr
import csv_reader as csvr
import npz_reader as npzr
import npy_reader as npyr
import data_displayer as dd
import io_tools as iot
import data_logger as dl
import naming_tools as nt
import experiment_runner as expr

def source_data(overwrite=False):
    dl.log("Started creationg folders based on config")
    folders_exist = iot.create_data_folders()
    if folders_exist:
        dl.log("All data folders exist...")
        dl.log("Started fuzzy data sourcing")
        iot.source_annotated_data_fuzzy(overwrite=overwrite)

def create_data_files(overwrite=True):
    dl.log("Started creating DLs")
    create_all_data(overwrite=overwrite)

def add_to_data(overwrite=True):
    dl.log("Started adding joystick data")
    write_joysticks_to_all_data(overwrite=overwrite)
    dl.log("Started adding facial feature data")
    write_facial_features_to_all_data(overwrite=overwrite)

def preprocess_data(overwrite=True):
    dl.log("Started cleaning data")
    clean_all_data(overwrite=overwrite)

def run_analysis(overwrite=True):
    dl.log("Started analysing data")
    summarize_all_data(overwrite=overwrite)

def run_statistics(overwrite=True, collapse=True):
    dl.log("Started gathering statistics")
    expr.run_statistics(overwrite=overwrite, collapse=collapse, groupings=[["SpeakerID"], ["holistic_cefr"], ["Score"]], tasks=["task5"])
    
def data_pipeline(overwrite=True):
    source_data(overwrite=overwrite)
    create_data_files(overwrite=overwrite)
    add_to_data(overwrite=overwrite)
    preprocess_data(overwrite=overwrite)
    run_analysis(overwrite=overwrite)
    run_statistics(overwrite=overwrite)
    dl.log("Finished processing")

def get_prints_data(key, row, wavs, overwrite=True):
    return [f"{key}, wavs: {len(wavs)}"]

def get_sanitized_tiers(key, row, wavs, overwrite=True):
    sanitized_tiers = []
    eaf = row["eaf"]
    tiers = list(eaf.tiers)
    for tier in tiers:
        sanitized_tier, _ = eafr.sanitize(tier)
        sanitized_tiers.append(sanitized_tier)
    return sanitized_tiers

def get_labelled_tier_labels(key, row, wavs, overwrite=True):
    labels = []
    eaf = row["eaf"]
    tiers = list(eaf.tiers)
    for tier in tiers:
        sanitized_tier, _ = eafr.sanitize(tier)
        if sanitized_tier in eafr.LABELLED_TIERS:
            for t0, t1, text in eaf.get_annotation_data_for_tier(tier):
                label = None
                if sanitized_tier == "text":
                    label = eafr.find_text_tokens(eafr.sanitize_text(text), nontextual_tokens="contains")
                else:
                    label = text
                label = eafr.sanitize_label(label)
                labels.append(sanitized_tier + " " + label)
    return labels

def create_data_matrix(key, row, wavs, overwrite=True):
    t_max = wavr.find_wavs_t_max(wavs)
    eaf = row['eaf']
    name = row['eafpath'].name
    annotation_data, annotation_labels = eafr.eaf_to_data_matrix(eaf, width=t_max, name=name)
    extraction_data, extraction_labels = wavr.wavs_to_data_matrix(wavs, t_max=t_max)

    if np.size(extraction_labels) > 0 and np.size(extraction_data) > 0:
        D = np.concatenate([extraction_data, annotation_data], axis=1).T
        L = np.concatenate([extraction_labels, annotation_labels])
    else:
        D = annotation_data.T
        L = annotation_labels
    
    return D, L

def create_and_display_DL(key, row, wavs, overwrite=True):
    D, L = create_data_matrix(key, row, wavs)
    dd.cc(D, L, key, overwrite=overwrite)
    return [key]

def create_and_write_DL(key, row, wavs, overwrite=True):
    D, L = create_data_matrix(key, row, wavs)
    _, res = npzr.write_DL(key, D, L, overwrite=overwrite)
    return res

def write_feature_to_data(name, D, L, method, overwrite=False):
    start_shape = D.shape
    # TODO: (low prio) (similar functionality exists) Implement overwrite for csvr methods
    name, D, L = method(name, D, L)
    D, L = npzr.reorder_data(D, L)
    npz_path, _ = npzr.write_DL(name, D, L, write_to_manifest=False, overwrite=True)
    end_shape = D.shape

    return [dl.write_to_manifest_file_change("data_matrix", npz_path, start_shape, end_shape, f"{method.__name__}")]

def write_joystick_to_data(name, D, L, overwrite):
    method = csvr.add_joysticks_to_data
    return write_feature_to_data(name, D, L, method, overwrite=overwrite)

def write_facial_feature_to_data(name, D, L, overwrite):
    method = csvr.add_facial_features_to_data
    return write_feature_to_data(name, D, L, method, overwrite=overwrite)

def clean_data(name, D, L, overwrite):
    res = npzr.clean_DL(name, D, L, overwrite=overwrite)
    return res

def summarize_data(name, D, L, overwrite):
    res = npyr.summarize_data(name, D, L, overwrite=overwrite)
    return res

def display_data(name, D, L, overwrite):
    dd.cc(D, L, name, overwrite=overwrite)
    return [name]

def get_prints_data(name, D, L, overwrite=False):
    return [f"{name}"]

def create_all_data(overwrite=True):
    res = iterate_through_data_provider(create_and_write_DL, overwrite=overwrite)

def display_all_data(overwrite = True):
    iterate_through_npz_provider(display_data, overwrite=overwrite)

def write_joysticks_to_all_data(overwrite = True):
    res = iterate_through_npz_provider(write_joystick_to_data, overwrite=overwrite)

def write_facial_features_to_all_data(overwrite = True):
    res = iterate_through_npz_provider(write_facial_feature_to_data, overwrite=overwrite)

def clean_all_data(overwrite=True):
    res = iterate_through_npz_provider(clean_data, overwrite=overwrite)

def summarize_all_data(overwrite=True):
    res = iterate_through_npz_provider(summarize_data, overwrite=overwrite)

# TODO: (low prio) make it so when not overwriting, files aren't read unnecassarily
def iterate_through_data_provider(method, n = math.inf, offset = 0, overwrite = True):
    i = 0
    aggregate = []
    provider = dr.DataProvider(sr = 16000)
    for keys, rows, wavss in provider.generator(do_wavs = True):
        if i >= n:
            break
        if i < offset:
            i += 1
            continue
        for key, row, wavs in zip(keys, rows, wavss):
            try:
                results = method(key, row, wavs, overwrite = overwrite)
                aggregate.extend(results)
            except Exception as e:
                dl.log(f"Failure at {i}:{method.__name__} - {key}\n\t{e}")
                dl.log_stack()
        i += 1
    return Counter(aggregate), i

def iterate_through_npz_provider(method, n = math.inf, offset = 0, overwrite = True):
    i = 0
    aggregate = []
    provider = npzr.NpzProvider()
    for names, Ds, Ls in provider.generator():
        if i >= n:
            break
        if i < offset:
            i += 1
            continue
        for name, D, L in zip(names, Ds, Ls):
            try:
                results = method(name, D, L, overwrite)
                aggregate.extend(results)
            except Exception as e:
                dl.log(f"Failure at {i}:{method.__name__} - {name}\n\t{e}")
                dl.log_stack()
        i += 1
    return Counter(aggregate), i

if __name__ == '__main__':
    globals()[sys.argv[1]]()
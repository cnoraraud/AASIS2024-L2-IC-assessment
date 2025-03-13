import sys
import math
import traceback
import eaf_consumer as ec
import npz_reader as npzr
import data_displayer as dd
import data_reader as dr
import numpy as np
import wav_consumer as wc
import csv_consumer as csvc
import npz_reader as npzr
import npy_reader as npy
import io_tools as iot

from collections import Counter

def get_prints_data(key, row, wavs):
    return [f"{key}, wavs: {len(wavs)}"]

def get_prints_data(name, D, L):
    return [f"{name}"]

def get_sanitized_tiers(key, row, wavs):
    sanitized_tiers = []
    eaf = row["eaf"]
    tiers = list(eaf.tiers)
    for tier in tiers:
        sanitized_tier, sanitized_number = ec.sanitize(tier)
        sanitized_tiers.append(sanitized_tier)
    return sanitized_tiers

def get_labelled_tier_labels(key, row, wavs):
    labels = []
    eaf = row["eaf"]
    tiers = list(eaf.tiers)
    for tier in tiers:
        sanitized_tier, sanitized_number = ec.sanitize(tier)
        if sanitized_tier in ec.LABELLED_TIERS:
            for t0, t1, text in eaf.get_annotation_data_for_tier(tier):
                label = None
                if sanitized_tier == "text":
                    label = ec.find_text_tokens(ec.sanitize_text(text), nontextual_tokens="contains")
                else:
                    label = text
                label = ec.sanitize_label(label)
                labels.append(sanitized_tier + " " + label)
    return labels

def create_all_data():
    res = iterate_through_data_provider(create_and_write_DLs)
    print(res)

def summarize_all_data():
    res = iterate_through_npz_provider(summarize_data)
    print(res)

def create_data(key, row, wavs):
    ms = wc.find_wavs_ms(wavs)
    extraction_data, extraction_labels = wc.wavs_to_ms(wavs)
    eaf = row['eaf']
    name = row['eafpath'].name
    annotation_data, annotation_labels = ec.process_eaf(eaf, ms, name=name)
    D = np.concatenate([extraction_data, annotation_data], axis=1).T
    L = np.concatenate([ec.sanitize_labels(extraction_labels, tag="(ext.)"), ec.sanitize_labels(annotation_labels, tag="(ann.)")])
    return D, L

def data_pipeline():
    print("Started creationg folders based on config")
    folders_exist = iot.create_data_folders()
    if folders_exist:
        print("All data folders exist...")
        print("Started fuzzy data sourcing")
        iot.source_annotated_data_fuzzy()
    print("Started creating DLs")
    create_all_data()
    print("Started adding joystick data")
    write_joysticks_to_all_data()
    print("Started adding facial feature data")
    write_facial_features_to_all_data()
    print("Started analysing data")
    summarize_all_data()
    print("Finished processing")

def get_display(key, row, wavs):
    D, L = create_data(key, row, wavs)
    dd.display_all(D, L, key)
    return [key]

def display_data(name, D, L):
    dd.display_all(D, L, name)
    return [name]

#TODO: Manifest this as well...
def create_and_write_DLs(key, row, wavs):
    D, L = create_data(key, row, wavs)
    filename = npzr.write_DL(key, D, L)
    return [filename]

def summarize_data(name, D, L):
    filename = npy.summarize_data(name, D, L)
    return [filename]

def iterate_through_data_provider(method, n = math.inf, offset = 0):
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
                results = method(key, row, wavs)
                aggregate.extend(results)
            except Exception as e:
                print(f"Failure at {i}: {key}   {e}")
                print(traceback.format_exc())
        i += 1
    return Counter(aggregate), i

def write_feature_to_data(name, D, L, method):
    start_shape = D.shape
    name, D, L = method(name, D, L)
    D, L = npzr.reorder_data(D, L)
    path = npzr.write_DL(name, D, L)
    end_shape = D.shape
    return [f"{path} shape: {start_shape} => {end_shape}"]

def write_joystick_to_data(name, D, L):
    method = csvc.add_joysticks_to_data
    return write_feature_to_data(name, D, L, method)

def write_facial_feature_to_data(name, D, L):
    method = csvc.add_facial_features_to_data
    return write_feature_to_data(name, D, L, method)

def display_all_data():
    iterate_through_npz_provider(display_data)

def write_joysticks_to_all_data():
    res = iterate_through_npz_provider(write_joystick_to_data)
    print(res)

def write_facial_features_to_all_data():
    res = iterate_through_npz_provider(write_facial_feature_to_data)
    print(res)

def iterate_through_npz_provider(method, n = math.inf, offset = 0):
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
                results = method(name, D, L)
                aggregate.extend(results)
            except Exception as e:
                print(f"Failure at {i}: {name}   {e}")
                print(traceback.format_exc())
        i += 1
    return Counter(aggregate), i

if __name__ == '__main__':
    globals()[sys.argv[1]]()
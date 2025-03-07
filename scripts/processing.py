import sys
import math
import eaf_consumer as ec
import npz_reader as npzr
import data_displayer as dd
import data_reader as dr
import numpy as np
import wav_consumer as wc
import csv_consumer as csvc
import npz_reader as npzr
import npy_reader as npy
import traceback

from collections import Counter

def get_prints_data(key, row, wavs):
    return [f"{key}, wavs: {len(wavs)}"]

def get_prints_npz(name, D, L):
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

def create_all_DLs():
    res = iterate_through_data_provider(create_and_write_DLs)
    print(res)

def analyze_all_npzs():
    res = iterate_through_npz_provider(analyze_npz)
    print(res)

def create_DL(key, row, wavs):
    ms = wc.find_wavs_ms(wavs)
    extraction_data, extraction_labels = wc.wavs_to_ms(wavs, ms)
    eaf = row['eaf']
    annotation_data, annotation_labels, annotations = ec.process_eafs(eaf, ms)
    D = np.concatenate([extraction_data, annotation_data], axis=1).T
    L = np.concatenate([ec.sanitize_labels(extraction_labels, tag="(ext.)"), ec.sanitize_labels(annotation_labels, tag="(ann.)")])
    return D, L

def do_all_processing():
    print("Started creating DLs")
    create_all_DLs()
    print("Started adding joystick data")
    write_joysticks_to_all_npzs()
    print("Started analysing data")
    analyze_all_npzs()
    print("Finished processing")

def get_display(key, row, wavs):
    D, L = create_DL(key, row, wavs)
    dd.display_all(D, L, key)
    return [key]

def display_npz(name, D, L):
    dd.display_all(D, L, name)
    return [name]

#TODO: Manifest this as well...
def create_and_write_DLs(key, row, wavs):
    D, L = create_DL(key, row, wavs)
    filename = npzr.write_DL(key, D, L)
    return [filename]

def analyze_npz(name, D, L):
    filename = npy.analyse_npz(name, D, L)
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

def write_joystick_to_npz(name, D, L):
    start_shape = D.shape
    name, D, L = csvc.get_joysticks_for_D(name, D, L)
    D, L = npzr.reorder_data(D, L)
    path = npzr.write_DL(name, D, L)
    end_shape = D.shape
    return [f"{path} shape: {start_shape} => {end_shape}"]

def display_all_npzs():
    iterate_through_npz_provider(display_npz)

def write_joysticks_to_all_npzs():
    res = iterate_through_npz_provider(write_joystick_to_npz)
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
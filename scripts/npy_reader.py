from datetime import datetime
import numpy as np
import npz_reader as npzr
import summary_reader as sumr
import io_tools as iot
import naming_tools as nt
import analysis as ana
import filtering as filt
import data_logger as dl

def get_summary_info(summary):
    individual_valid, individual_total, relational_valid, relational_total, valid_structure = sumr.count_label_values(summary)
    features = summary["general"]["features"]
    performance_length = summary["general"]["performance_length"]
    input_info = f"({features},{performance_length})"
    if valid_structure:
        individual_valid_ratio = individual_valid/individual_total
        relational_valid_ratio = relational_valid/relational_total
        output_info = f"{individual_total} ({individual_valid_ratio:.2}) {relational_total} ({relational_valid_ratio:.2})"
    else:
        output_info = f"invalid summary structue"
    return input_info, output_info

def write_summary(name, summary):    
    npz_path = iot.npzs_path() / nt.file_swap(name, "npz", all=False)
    npy_path = iot.npys_path() / nt.file_swap(name, "npy", all=False)
    input_info, output_info = get_summary_info(summary)
    np.save(str(npy_path), summary)
    return [dl.write_to_manifest_new_file("summary", npz_path, npy_path, input_info=input_info, output_info=output_info)]
        
def summarize_data(name, D, L):
    L = npzr.anonymize_speakers(L)
    D_focused, L = npzr.focus_on_label(D, L, "performance")
    times, starting = ana.turn_taking_times_comparative(D, L, n=10000)
    L_filter = npzr.canonical_select(L, [["has","ic:"],[("has","ann.")],[("has","energy")]]) & npzr.double_speaker_filter(L)
    D_focused, L = npzr.do_label_select(D_focused, L, L_filter)
    D_flat = filt.flatten(D_focused)
    ond_analyses = ana.get_all_segment_overlaps_and_delays(L, D_flat)
    mnw_analyses = ana.get_all_segment_masses_and_widths(L, D_focused)
    corr_analyses = ana.get_all_corellations(L, D_focused)
    analyses = ana.group_analyses(L, ond_analyses, mnw_analyses, corr_analyses)
    
    summary = dict()
    summary = {"label_summaries": ana.summarize_analyses(L, analyses),
               "general": {
                    "features": D_focused.shape[0],
                    "length": D.shape[1],
                    "performance_length": D_focused.shape[1]},
                "labels": L,
                "name": name}

    write_summary(name, summary)

def npy_list():
    npys = []
    for name in iot.list_dir(iot.npys_path(), "npy"):
        npys.append(name)
    return npys

def load_summary_from_name(name):
    name = nt.file_swap(name, "npy")
    return load_summary(iot.npys_path() / name)

def load_summary(name):
    summary = np.load(name,allow_pickle=True).item()
    return name, summary

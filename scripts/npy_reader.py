import os
import time
from datetime import datetime
import io_tools as iot
import naming_tools as nt
import npz_reader as npzr
import analysis as ana
import numpy as np
import filtering as filt

def write_summary(name, summary):
    npys_path = iot.npys_path()
    with open(npys_path / "analysis_manifest.txt", "a") as manifest:
        today = datetime.now()
        npz_name, modified = npzr.read_DL_metadata_from_name(name)
        npy_name = nt.file_swap(name, "npy")
        np.save(str(npys_path / npy_name), summary) 
        manifest.write(f"{npz_name} ({modified}) => {npy_name} ({today})\n")

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
    summary["label_summaries"] = ana.summarize_analyses(L, analyses)
    summary["general"] = {
        "features": L.shape[0],
        "length": D.shape[1],
        "performance_length": D_focused.shape[1],
        }
    summary["labels"] = L

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

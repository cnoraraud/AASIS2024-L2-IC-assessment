import os
import time
from datetime import datetime
import io_tools as iot
import npz_reader as npzr
import analysis as ana
import numpy as np

def analyse_npz(name, D, L):
    speakers = npzr.find_speakers_from_name(name)
    L_anon = npzr.anonymize_speakers(L, speakers)
    D_s, L_anon = npzr.focus_on_label(D, L_anon, "performance")
    L_filter = npzr.canonical_select(L_anon, [[("has","ann.")],[("has","energy")]]) & npzr.double_speaker_filter(L_anon)
    D_s, L_anon = npzr.do_label_select(D_s, L_anon, L_filter)
    Dflat = npzr.flatten_data(D_s)
    ond_analyses = ana.get_all_segment_overlaps_and_delays(L_anon, Dflat)
    mnw_analyses = ana.get_all_segment_masses_and_widths(L_anon, D_s)
    corr_analyses = ana.get_all_corellations(L_anon, D_s)
    analyses = ana.group_analyses(L_anon, ond_analyses, mnw_analyses, corr_analyses)
    
    summary = dict()
    summary["label_summaries"] = ana.summarize_analyses(L_anon, analyses)
    summary["general"] = {"features": D.shape[0], "length": D.shape[1], "performance_length": D_s.shape[1]}
    summary["labels"] = L_anon
    npys_path = iot.npys_path()
    with open(npys_path / "analysis_manifest.txt", "a") as manifest:
        today = datetime.now()
        npy_name = str(npys_path / name)
        if ".npy" not in npy_name:
            npy_name = npy_name + ".npy"
        npz_name, modified = npzr.read_DL_metadata_from_name(name)
        np.save(npy_name, summary) 
        manifest.write(f"{npz_name} ({modified}) => {npy_name} ({today})\n")

def npy_list():
    npys = []
    for name in iot.list_dir(iot.npys_path(), "npy"):
        npys.append(name)
    return npys

def load_summary_from_name(name):
    name = name.replace(".npz",".npy")
    if ".npy" not in name:
        name = name + ".npy"
    return load_summary(iot.npys_path() / name)

def load_summary(name):
    summary = np.load(name,allow_pickle=True).item()
    return name, summary

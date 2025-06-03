from datetime import datetime
import numpy as np
import npz_reader as npzr
import summary_reader as sumr
import io_tools as iot
import naming_tools as nt
import analysis as ana
import filtering as filt
import data_logger as dl

def get_summary_info_io(summary):
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
    input_info, output_info = get_summary_info_io(summary)
    np.save(str(npy_path), summary)
    del summary
    return [dl.write_to_manifest_new_file(dl.SUMMARY_TYPE, npz_path, npy_path, input_info=input_info, output_info=output_info)]
        
def summarize_data(name, D, L, overwrite=False):
    if not overwrite:
        npy_path = iot.npys_path() / nt.file_swap(name, "npy", all = False)
        if npy_path.exists():
            existed_log = f"{npy_path} existed, skipping summary step."
            dl.tprint(existed_log)
            return [existed_log]

    L = npzr.anonymize_speakers(L)
    D_focused, L = npzr.focus_on_label(D, L, "performance")
    D_focused, L = npzr.add_turntaking(D_focused, L)
    
    segment_filter = npzr.has(L, nt.AA_TAG) | npzr.has(L, nt.ANNOTATION_TAG) | npzr.has(L, "vad")
    ond_filter = segment_filter
    mnw_filter = segment_filter
    corr_filter = npzr.full(L)

    
    D_ond, L_ond = npzr.do_label_select(D_focused, L, ond_filter)
    D_ond = filt.flatten(D_ond)
    ond_analyses = ana.get_all_segment_overlaps_and_delays(L_ond, D_ond)

    D_mnw, L_mnw = npzr.do_label_select(D_focused, L, mnw_filter)
    mnw_analyses = ana.get_all_segment_masses_and_widths(L_mnw, D_mnw)

    D_corr, L_corr = npzr.do_label_select(D_focused, L, corr_filter)
    corr_analyses = ana.get_all_correlations(L_corr, D_corr)

    del D_ond, D_mnw, D_corr
    
    analyses = ana.group_analyses(L, ond_analyses, mnw_analyses, corr_analyses)
    Ls = {
        "overlaps_and_delays": L_ond,
        "masses_and_widths": L_mnw,
        "correlations": L_corr
    }

    del ond_analyses, mnw_analyses, corr_analyses

    summary = dict()
    summary = {"label_summaries": ana.summarize_analyses(L, analyses, Ls),
               "general": {
                    "features": D_focused.shape[0],
                    "length": D.shape[1],
                    "performance_length": D_focused.shape[1]},
                "labels": L,
                "name": name}
    
    del Ls, L_ond, L_mnw, L_corr, analyses

    return write_summary(name, summary)

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

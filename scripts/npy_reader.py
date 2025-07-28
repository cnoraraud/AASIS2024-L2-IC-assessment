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
    features = summary["general"]["feature_count"]
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
    input_info = None
    output_info = None

    np.save(str(npy_path), summary)
    try:
        input_info, output_info = get_summary_info_io(summary)
    except Exception as e:
        dl.log_stack("Failed to create summary info.")
    
    del summary
    return [dl.write_to_manifest_new_file(dl.SUMMARY_TYPE, npz_path, npy_path, input_info=input_info, output_info=output_info)]

def get_summary(name, D, L):
    L = npzr.anonymize_speakers(L)
    
    D_focused, L = npzr.focus_on_label(D, L, "performance")
    D_focused, L = npzr.add_turntaking(D_focused, L)

    feature_count = D_focused.shape[0]
    performance_length = D_focused.shape[1]
    total_length = D.shape[1]

    D_smooth = npzr.get_D_smooth(D_focused)
    discrete_channels, binary_channels = npzr.identify_data_type(D_focused, L)
    
    D_binary = npzr.transform_to_binary(D_focused, discrete_channels, binary_channels, D_smooth=D_smooth)
    D_thresholded = npzr.transform_to_thresholded(D_focused, discrete_channels, binary_channels, D_smooth=D_smooth)
    D_continuous = npzr.transform_to_continuous(D_focused, discrete_channels, binary_channels, D_smooth=D_smooth)

    L_sc = L
    L_mnw = L
    L_corr = L

    ond_analyses = ana.get_all_segment_comparisons(L_sc, D_binary, D_thresholded)

    mnw_analyses = ana.get_all_segment_masses_and_widths(L_mnw, D_thresholded)

    corr_analyses = ana.get_all_correlations(L_corr, D_continuous)

    del D_binary, D_thresholded, D_continuous, D_focused, D_smooth
    
    analyses = ana.group_analyses(L, ond_analyses, mnw_analyses, corr_analyses)
    Ls = {
        "segment_comparisons": L_sc,
        "masses_and_widths": L_mnw,
        "correlations": L_corr
    }

    del ond_analyses, mnw_analyses, corr_analyses

    summary = dict()
    summary = {"label_summaries": ana.summarize_analyses(L, analyses, Ls),
               "general": {
                    "feature_count": feature_count,
                    "length": total_length,
                    "performance_length": performance_length},
                "labels": L,
                "name": name}
    
    del Ls, L_sc, L_mnw, L_corr, analyses
    
    return summary

def summarize_data(name, D, L, overwrite=False):
    if not overwrite:
        npy_path = iot.npys_path() / nt.file_swap(name, "npy", all = False)
        if npy_path.exists():
            existed_log = f"{npy_path} existed, skipping summary step."
            dl.log(existed_log)
            return [existed_log]
    summary = get_summary(name, D, L)
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

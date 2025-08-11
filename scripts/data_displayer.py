import math
import sys

import analysis as ana
import data_logger as dl
import eaf_reader as eafr
import filtering as filt
import io_tools as iot
import matplotlib.pyplot as plt
import naming_tools as nt
import npz_reader as npzr
import numpy as np
import numpy_wrapper as npw


def sanitize_labels(labels, tag=""):
    labels = npw.string_array(labels)
    sanitized_labels = np.empty(labels.shape, dtype = labels.dtype)
    for i, label in np.ndenumerate(labels):
        sanitized_label, sanitized_number = eafr.sanitize(label)
        if sanitized_number is None:
            sanitized_number = "[all]"
        tagspace = " "
        if len(tag) > 0:
            tagspace = " " + tag + " "
        sanitized_labels[i] = f"{sanitized_number}{tagspace}{sanitized_label}"
    return sanitized_labels

def save_figure(title, subfolder=None, overwrite=True):
    safe_title = nt.sanitize_filename(title)
    savepath = iot.figs_path()
    if not isinstance(subfolder, type(None)):
        savepath = iot.figs_path() / f"{subfolder}"
    iot.create_missing_folder_recursive(savepath)
    fullsavepath = savepath / f"{safe_title}.png"
    if overwrite or not fullsavepath.exists():
        plt.savefig(fullsavepath, dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()

def produce_side_by_side_speaker_plot(data, labels, title, rescale_rows=True, zero_mode=False, reorder=True, max_t=None, style="discrete", norm="symlog", colorbar=False, savefig=False, overwrite=True, subfolder=None, time=None, dominutelines=False):
    labels = npzr.anonymize_speakers(labels)
    relationship_filter = npzr.has(labels, nt.ANNOTATION_TAG) & ~npzr.has(labels, nt.SPEAKERS)
    S1_filter = npzr.has(labels, nt.SPEAKER1) & relationship_filter
    S2_filter = npzr.has(labels, nt.SPEAKER2) & relationship_filter
    D1, L1  = npzr.do_label_select(data, labels, S1_filter)
    D2, L2 = npzr.do_label_select(data, labels, S2_filter)
    L1 = npzr.strip_label_slots(L1)
    L2 = npzr.strip_label_slots(L2)
    L_common = npw.string_array(sorted(list(set(L1.tolist()) | set(L2.tolist()))))
    L1_incommon = np.isin(L_common, L1)
    L2_incommon = np.isin(L_common, L2)
    D_common_shape = (L_common.shape[0], data.shape[1])
    D1_common = np.full(D_common_shape, np.nan)
    D2_common = np.full(D_common_shape, np.nan)
    D1_common[L1_incommon, :] = D1
    D2_common[L2_incommon, :] = D2

    plot_labels = L_common
    if npw.is_string(plot_labels):
            plot_labels = [plot_labels]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(15)
    fig.set_figheight(len(L_common)/4)
    fig.suptitle(f"{title}", fontsize=16)
    fig.supxlabel('Time (ms)')
    fig.supylabel('Feature')
    
    if reorder:
        D1_common, L_common = npzr.reorder_data(D1_common, L_common)
        D2_common, L_common = npzr.reorder_data(D2_common, L_common)

    cc(D1_common, L_common, nt.SPEAKER1, fig=fig, ax=ax1, reorder=False, rescale_rows=rescale_rows, zero_mode=zero_mode, max_t=max_t, style=style, norm=norm, colorbar=colorbar, time=time, dominutelines=dominutelines)
    cc(D2_common, L_common, nt.SPEAKER2, fig=fig, ax=ax2, reorder=False, rescale_rows=rescale_rows, zero_mode=zero_mode, max_t=max_t, style=style, norm=norm, colorbar=colorbar, time=time, dominutelines=dominutelines)
    if savefig:
        save_figure(title, subfolder=subfolder, overwrite=overwrite)
    else:
        plt.show()

def cc(data, labels, title, reorder=True, rescale_rows=False, zero_mode=False, max_t=None, style="discrete", norm="symlog", colorbar=False, fig=None, ax=None, savefig=False, overwrite=True, subfolder=None, time=None, dominutelines=False):
    individual = fig is None or ax is None
    cmap = None
    if style == "discrete":
        cmap = "nipy_spectral"
    elif style == "uniform":
        cmap = "gist_heat"
    elif style == "symmetrical":
        cmap = "berlin"
    else:
        cmap = style
    plot_data = data
    plot_labels = labels
    if isinstance(rescale_rows, tuple) or isinstance(rescale_rows, list):
        vmin = rescale_rows[0]
        vmax = rescale_rows[1]
    elif rescale_rows == "0_centre":
        mag = np.max(np.abs(plot_data))
        vmin = -mag
        vmax = mag
    elif rescale_rows == "01":
        plot_data = filt.to_01(plot_data)
        vmin = 0
        vmax = 1
    elif rescale_rows == "01_both":
        pos_mask = plot_data >= 0
        neg_mask = plot_data < 0
        plot_data_pos = np.copy(plot_data)
        plot_data_pos[plot_data_pos < 0] = 0
        plot_data_pos = filt.to_01(plot_data_pos)
        plot_data_neg = -np.copy(plot_data)
        plot_data_neg[plot_data_neg < 0] = 0
        plot_data_neg = -filt.to_01(plot_data_neg)
        plot_data = np.zeros_like(plot_data)
        plot_data[pos_mask] = plot_data_pos[pos_mask]
        plot_data[neg_mask] = plot_data_neg[neg_mask]
        vmin = -1
        vmax = 1
    elif rescale_rows == "asym":
        plot_data = filt.norm(plot_data)
        vmin = -1
        vmax = 1
    elif rescale_rows == "sym":
        plot_data = np.abs(filt.norm(plot_data))
        vmin = 0
        vmax = 1
    else:
        vmin = np.nanmin(plot_data)
        vmax = np.nanmax(plot_data)
    if zero_mode:
        plot_data = filt.to_0_mode(plot_data)
        vmin = np.nanmin(plot_data)
        vmax = np.nanmax(plot_data)        
    if not isinstance(max_t, type(None)):
        plot_data = ana.apply_method_to_all_features(plot_data, filt.fit_to_size, {"t":max_t, "destructive":True})
    if reorder:
        plot_data, plot_labels = npzr.reorder_data(plot_data, plot_labels)
    if npw.is_string(plot_labels):
        plot_labels = [plot_labels]
    if individual:
        fig, (ax) = plt.subplots(1, 1)
        fig.set_figwidth(20)
        fig.set_figheight(len(plot_labels)/4)
    cax = ax.imshow(np.atleast_2d(plot_data), cmap=cmap, aspect='auto', interpolation='none', norm=norm, vmin=vmin, vmax=vmax)
    if colorbar:
        colorbar = "vertical"
    if isinstance(colorbar, str):
        fig.colorbar(cax, orientation=colorbar, aspect=round(1*len(plot_labels)))
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=True, right=False, labelright=False)
    ax.set_title(title)
    if len(plot_labels) > 0:
        ax.set_yticks(np.arange(len(plot_labels)), labels = plot_labels)
    if time == "mstos":
        x_lims = ax.get_xlim()
        labels = ax.get_xticklabels()
        new_labels = []
        new_values = []
        for i in range(len(labels)):
            val = int(float(labels[i].get_text().replace("−","-")))
            newval = int(round(val/1000))
            new_labels.append(f"{newval}".replace("-","−"))
            new_values.append(labels[i].get_position()[0])
        ax.set_xticks(new_values)
        ax.set_xticklabels(new_labels)
        ax.set_xlim(x_lims)
    elif time == "mstoms":
        x_lims = ax.get_xlim()
        new_labels = []
        new_values = []
        minute_lines = []
        freq = 30000
        for i in range(int(x_lims[0]) + freq, int(x_lims[1]), freq):
            new_values.append(i)
            seconds = int(round(i/1000))
            minutes = seconds//60
            seconds = seconds - minutes*60
            if seconds == 0:
                minute_lines.append(i)
            minutes_text = f"{minutes}".rjust(2,"0")
            seconds_text = f"{seconds}".rjust(2,"0")
            new_labels.append(f"{minutes_text}:{seconds_text}")
        if dominutelines:
            ax.vlines(minute_lines, 0, 1, transform=ax.get_xaxis_transform(), linewidth=0.5, color='cyan')
        ax.set_xticks(new_values)
        ax.set_xticklabels(new_labels)
        ax.set_xlim(x_lims)
    elif time == "none":
        ax.set_xticks([])
        ax.set_xticklabels([])
    if individual:
        if savefig:
            save_figure(title, subfolder=subfolder, overwrite=overwrite)
        else:
            plt.show()

def max_string_length(labels):
    max_length = 0
    for label in labels:
        max_length = max(max_length, len(label))
    return max_length

def label_rotation(max_str_length, width, count):
    w_ratio = (max_str_length - width) / width
    c_ratio = count*5
    ratio_angle = w_ratio * c_ratio
    clipped_angle = max(0,min(90, ratio_angle))
    snap = 15
    return round(clipped_angle / snap) * snap

def r(data, title, labels, labels_top=None, colorbar=False, vmin=None, width=15, height=15, vmax=None, fig = None, ax = None, labeltop = True, labelleft = True, savefig=False, overwrite=True, automate_colorscale=True, zero_centre=False, subfolder=None, annotate=False, normalize=None, y_label=None, x_label=None):
    individual = fig is None or ax is None
    if individual:
        fig, (ax) = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
    if isinstance(labels_top, type(None)):
        labels_top = labels
    if normalize == "y":
        data = data/data.sum(axis=0, keepdims=True)
    elif normalize == "x":
        data = data/data.sum(axis=1, keepdims=True)
    elif normalize == "all":
        data = data/data.sum(keepdims=True)
    data = np.nan_to_num(data)

    symmetric = False
    vmin = data.min()
    vmax = data.max()
    if automate_colorscale:
        new_data = np.array(data)
        if zero_centre:
            centre = 0
        else:
            centre = np.nanmedian(new_data)
        distance = abs(new_data - centre)
        new_data[distance > 3 * np.std(new_data)] = np.nan
        centre = np.nanmedian(new_data)
        mag = np.nanstd(new_data) * 3
        top = np.nanmax(new_data)
        bot = np.nanmin(new_data)
        if isinstance(vmin, type(None)):
            vmin = max(centre - mag, bot)
        if isinstance(vmax, type(None)):
            vmax = min(centre + mag, top)
        if abs(centre) < np.finfo(np.float32).eps:
            symmetric = True
        elif abs(centre) < abs(mag):
            symmetric = abs(centre) / abs(mag) < 0.2
    cmap = "plasma"
    cols = ["white","black"]
    if symmetric:
        cmap = "berlin"
        cols = ["white","black"]
    cax = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='none', vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(cax)
    ax.tick_params(top=False, labeltop=labeltop, bottom=False, labelbottom=False, left=False, labelleft=labelleft, right=False, labelright=False)
    x_pos = np.arange(len(labels_top))
    y_pos = np.arange(len(labels))
    ax.set_xticks(x_pos)
    ax.set_yticks(y_pos)
    rotation = label_rotation(max_string_length(labels_top), width, len(labels_top))
    ax.set_xticklabels(labels_top, rotation=rotation, ha='left')
    ax.set_yticklabels(labels)
    if annotate:
        for i in y_pos:
            for j in x_pos:
                c_i = 0
                if symmetric:
                    if data[i, j] > vmax/2 or data[i, j] < vmin/2:
                        c_i = 1
                else:
                    if data[i, j] > vmax/2:
                        c_i = 1
                val = data[i, j]
                if isinstance(val, float):
                    val_text = f"{val:.2f}"
                else:
                    val_text = f"{val}"
                ax.text(j, i, val_text, ha="center", va="center", color=cols[c_i])
    if npw.is_string(x_label):
        ax.set_xlabel(x_label)
    if npw.is_string(y_label):
        ax.set_ylabel(y_label)  
    ax.set_title(title)
    if individual:
        if savefig:
            save_figure(title, subfolder=subfolder, overwrite=overwrite)
        else:
            plt.show()

def s(x, datas, title, save_fig=False, x_label=None, y_label=None, width=15, height=15, fig = None, ax = None, xlim=None, ylim=None, overwrite=True, subfolder=None, savefig=False):
    individual = fig is None or ax is None
    if individual:
        fig, (ax) = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
    
    if not isinstance(datas, dict):
        datas = {"data": {"y": datas, "c":"blue"}}
    
    for data_key in datas:
        data = datas[data_key]
        y = data["y"]
        c = data["c"]
        label = None
        if "name" in data:
            label = data["name"]
        ax.scatter(x, y, s=2, c=c, label=label)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if npw.is_string(x_label):
        ax.set_xlabel(x_label)
    if npw.is_string(y_label):
        ax.set_ylabel(y_label)  

    if individual:
        if savefig:
            save_figure(title, subfolder=subfolder, overwrite=overwrite)
        else:
            plt.show()
def plot_line(y, color, n=500, label="", smooth=False, x=None, alpha=1):
    if np.size(y) == 0:
        return
    if isinstance(x, type(None)):
        x = np.arange(y.shape[0])
    if smooth: 
        feature_smooth = filt.ma(y, {"n":n})
        feature_smooth = filt.ma(feature_smooth, {"n":n})
        feature_smooth = filt.ma(feature_smooth, {"n":n})
        plt.plot(x, feature_smooth, color = color, ls='--')
        alpha = 0.25
    plt.plot(x, y, color = color, alpha = alpha, label=label)

def round_to_mult(num, mult = 10):
    anchor = 1
    if num == 0:
        return 0
    while not np.isclose(anchor, abs(num)) and abs(num) < anchor:
        anchor = anchor/mult
    while not np.isclose(anchor, abs(num)) and abs(num) > anchor:
        anchor = anchor*mult
    return anchor

def round_to_closest_mult(num, mults=(2,4,8,5,10,20)):
    best_distance = math.inf
    best_round = num
    for mult in mults:
        mult_round = round_to_mult(num, mult)
        distance = abs(mult_round - abs(num))
        if distance < best_distance:
            best_distance = distance
            best_round = mult_round
    return best_round


def get_axes(matrix, labels, feature_name, do_lines=True):
    matching_labels = npzr.has(labels, feature_name)
    matched_labels = labels[npzr.has(labels, feature_name)]
    line_count = round(sum(matching_labels))
    unstacked_features = matrix[matching_labels,:,:]
    t = unstacked_features.shape[1]
    features = unstacked_features.transpose(1, 0, 2).reshape(t, -1)
    N = round(features.shape[-1]/line_count)
    features = np.nan_to_num(features)
    
    y_max = 0
    y_min = 0
    
    axes = []
    if do_lines:
        for i in range(line_count):
            label = " ".join(matched_labels[i].split(" ")[0:])
            feature = np.nanmean(features[:, i*N:(i+1)*N], axis=-1)
            y_max = max(y_max, np.max(feature))
            y_min = min(y_min, np.min(feature))
            axes.append({"y": feature, "label": label, "type": "subset", "sample_N": N})
    if line_count > 1 or not do_lines:
        label = "Combined"
        feature_means = np.nanmean(features, axis=-1)
        y_max = max(y_max, np.max(feature_means))
        y_min = max(y_min, np.min(feature_means))
        axes.append({"y": feature_means, "label": label, "type": "superset", "sample_N": N*line_count})

    return {"axes": axes, "N": len(axes), "sample_N": N, "t": t, "y_lim": (y_min, y_max), "feature_name": feature_name}

def plot_feature(plt_data, task_name, mid_point=None, save_fig=False, force_scale=False, x_is_ratio=False, overwrite=True, subfolder=None):
    # Find x axis
    x = np.arange(plt_data["t"]) - 1
    if np.isscalar(mid_point):
        x = x - mid_point
    if x_is_ratio:
        x = x / np.max(np.abs(x))
    
    draw_smooths = plt_data["sample_N"] < 30 or plt_data["N"] <= 3
    draw_grid = plt_data["t"] > 50000

    # Plots
    color = iter(plt.cm.rainbow(np.linspace(0, 1, plt_data["N"])))
    for ax in plt_data["axes"]:
        ax_color = "black"
        if ax["type"] == "subset":
            ax_color = next(color)
        if ax["type"] == "superset":
            ax_color = "black"
        y = ax["y"]
        label = ax["label"]
        plot_line(y, color=ax_color, label=f"{label}", smooth=draw_smooths, x=x)
    
    if np.isscalar(mid_point):
        plt.axvline(0, c = 'black', alpha=0.5, lw=1, label="Event Point")
    y_lim = plt_data["y_lim"]
    if force_scale:
        y_bot = min(0, y_lim[0])
        y_top = max(y_lim[1], 1)
        y_round_sym = max(round_to_closest_mult(y_bot), round_to_closest_mult(y_top))
        y_top = y_round_sym
        if y_bot < 0:
            y_bot = -y_round_sym
        plt.yticks(np.linspace(y_bot,y_top,11))
        plt.ylim(y_bot, y_top)
    if draw_grid:
        plt.grid(True, which='both', axis='y')
    plt.xlim(np.min(x), np.max(x))

    # Legend and Title
    feature_name = plt_data["feature_name"]
    title = f"Mean occurrences of \'{feature_name}\' in task \'{task_name}\'"
    N_string = f"(N={plt_data["sample_N"]})"
    modifiers = [N_string]
    if np.isscalar(mid_point):
        modifiers.insert(0, "on speaker change")
    title_modifiers = " ".join(modifiers)
    if len(modifiers) > 0:
        title = "\n".join([title, title_modifiers])
    plt.title(title)
    plt.legend()
    
    # Displaying / Saving
    if save_fig:
        plot_name = "mean-occurrence"
        if np.isscalar(mid_point):
            plot_name = "mean-occurence-turn-taking"
        title = f"{plot_name}_{task_name}_{feature_name}"
        save_figure(title, subfolder=subfolder, overwrite=overwrite)
    else:
        plt.show()



def get_plt_data_turn_taking(task_name, feature_name, n = 5000, use_density = False):
    # only looking at one feature at a time does not optimize for cpu time when working with a batch, but uses less ram
    total_segments = []
    for npz in npzr.DL_names():
        if task_name in npz:
            segments = npzr.get_turn_taking_data_segments(npz, [[("has",feature_name)]], n = n, use_density = use_density)
            total_segments += segments
    TL, TD, mid_point = npzr.combined_segment_matrix_with_anchor(total_segments)
    plt_data = get_axes(TD, TL, feature_name)
    del TL
    del TD
    return plt_data, mid_point

def get_plt_data_overall(task_name, feature_name):
    # only looking at one feature at a time does not optimize for cpu time when working with a batch, but uses less ram
    total_segments = []
    for npz in npzr.DL_names():
        if task_name in npz:
            segments = npzr.get_entire_data_as_segment(npz, [[("has",feature_name)]])
            total_segments += segments
    TL, TD = npzr.combined_segment_matrix_with_interpolation(total_segments)
    plt_data = get_axes(TD, TL, feature_name, do_lines=False)
    del TL
    del TD
    return plt_data

def produce_figures_turn_taking(task_name, feature_name, n = 5000, use_density = False):
    plt_data, mid_point = get_plt_data_turn_taking(task_name, feature_name, n = n, use_density = use_density)
    plot_feature(plt_data, task_name, mid_point = mid_point, save_fig=True, subfolder=f"turntaking\\{task_name}")

def produce_figures_overall(task_name, feature_name):
    plt_data = get_plt_data_overall(task_name, feature_name)
    plot_feature(plt_data, task_name, x_is_ratio=True, force_scale=True, save_fig=True, subfolder=f"overall\\{task_name}")

def produce_all_figures_turn_taking():
    channel_names = npzr.get_all_channels()
    for task_name in ["task5","task4A","task4B"]:
        for feature_name in channel_names:
            try:
                produce_figures_turn_taking(task_name, feature_name, n = 10000, use_density = False)
            except Exception as e:
                dl.log_stack(f"Failed to produce figure for {task_name} {feature_name}. ({e})")

def produce_all_figures_overall():
    channel_names = npzr.get_all_channels()
    for task_name in ["task5","task4A","task4B"]:
        for feature_name in channel_names:
            try:
                produce_figures_overall(task_name, feature_name)
            except Exception as e:
                dl.log_stack(f"Failed to produce figure for {task_name} {feature_name}. ({e})")

def produce_all_figures():
    produce_all_figures_turn_taking()
    produce_all_figures_overall()

if __name__ == '__main__':
    globals()[sys.argv[1]]()
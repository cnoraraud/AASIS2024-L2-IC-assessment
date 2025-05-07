import sys
import math
import traceback
import matplotlib.pyplot as plt
import numpy as np
import numpy_wrapper as npw
import eaf_reader as eafr
import npz_reader as npzr
import io_tools as iot
import naming_tools as nt
import filtering as filt
import analysis as ana

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

def produce_side_by_side_speaker_plot(data, labels, title, rescale_rows=True, reorder=True, max_t=None, style="discrete", norm="symlog", colorbar=False, savefig=False, overwrite=True):
    labels = npzr.anonymize_speakers(labels)
    relationship_filter = npzr.has(labels, nt.ANNOTATION_TAG) & ~npzr.has(labels, npw.SPEAKERS)
    S1_filter = npzr.has(labels, npw.SPEAKER1) & relationship_filter
    S2_filter = npzr.has(labels, npw.SPEAKER2) & relationship_filter
    D1, L1 = npzr.do_label_select(data, labels, S1_filter)
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

    cc(D1_common, L_common, npw.SPEAKER1, fig=fig, ax=ax1, reorder=False, rescale_rows=rescale_rows, max_t=max_t, style=style, norm=norm, colorbar=colorbar)
    cc(D2_common, L_common, npw.SPEAKER2, fig=fig, ax=ax2, reorder=False, rescale_rows=rescale_rows, max_t=max_t, style=style, norm=norm, colorbar=colorbar)
    if savefig:
        safe_title = nt.sanitize_filename(title)
        savepath = iot.figs_path() / f"{safe_title}.png"
        if overwrite or not savepath.exists():
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()
    else:
        plt.show()

def cc(data, labels, title, reorder=True, rescale_rows=True, max_t=None, style="discrete", norm="symlog", colorbar=False, fig=None, ax=None, savefig=False, overwrite=True):
    individual = fig == None or ax == None
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
    if rescale_rows == "01":
        plot_data = filt.to_01(plot_data)
        vmin = 0
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
    if colorbar: fig.colorbar(cax)
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=True, right=False, labelright=False)
    ax.set_title(title)
    if len(plot_labels) > 0:
        ax.set_yticks(np.arange(len(plot_labels)), labels = plot_labels)
    if individual:
        if savefig:
            safe_title = nt.sanitize_filename(title)
            savepath = iot.figs_path() / f"{safe_title}.png"
            if overwrite or not savepath.exists():
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
        else:
            plt.show()

def r(data, title, labels, labels_top=None, colorbar=False, vmin=None, width=15, height=15, vmax=None, fig = None, ax = None, labeltop = True, labelleft = True, savefig=False, overwrite=True, automate_colorscale=True):
    individual = fig == None or ax == None
    if individual:
        fig, (ax) = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
    if isinstance(labels_top, type(None)):
        labels_top = labels
    symmetric = False
    if automate_colorscale:
        new_data = np.array(data)
        new_data[abs(data - np.nanmedian(data)) > 3 * np.std(data)] = np.nan
        centre = np.nanmedian(new_data)
        mag = np.nanmedian(np.abs(new_data - centre)) * 3
        top = np.nanmax(data)
        bot = np.nanmin(data)
        if isinstance(vmin, type(None)):
            vmin = max(centre - mag, bot)
        if isinstance(vmax, type(None)):
            vmax = min(centre + mag, top)
        symmetric = centre / mag < 0.2
    cmap = "plasma"
    if symmetric:
        cmap = "berlin"
    cax = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='none', vmin=vmin, vmax=vmax)
    if colorbar: fig.colorbar(cax)
    ax.tick_params(top=False, labeltop=labeltop, bottom=False, labelbottom=False, left=False, labelleft=labelleft, right=False, labelright=False)
    ax.set_xticks(np.arange(len(labels_top)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels_top, rotation=90, ha='left')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    if individual:
        if savefig:
            safe_title = nt.sanitize_filename(title)
            savepath = iot.figs_path() / f"{safe_title}.png"
            if overwrite or not savepath.exists():
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
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

def round_to_closest_mult(num, mults=[2,4,8,5,10,20]):
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

def plot_feature(plt_data, task_name, mid_point=None, save_fig=False, force_scale=False, x_is_ratio=False, overwrite=True):
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
    modifiers = [f"(N={plt_data["sample_N"]})"]
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
        safe_title = nt.sanitize_filename(f"{plot_name}_{task_name}_{feature_name}")
        savepath = iot.figs_path() / f"{safe_title}.png"
        if overwrite or not savepath.exists():
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()
    else:
        plt.show()



def get_plt_data_turn_taking(task_name, feature_name, n = 5000, use_density = False):
    # only looking at one feature at a time does not optimize for cpu time when working with a batch, but uses less ram
    total_segments = []
    for npz in npzr.npz_list():
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
    for npz in npzr.npz_list():
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
    plot_feature(plt_data, task_name, mid_point = mid_point, save_fig=True)

def produce_figures_overall(task_name, feature_name):
    plt_data = get_plt_data_overall(task_name, feature_name)
    plot_feature(plt_data, task_name, x_is_ratio=True, force_scale=True, save_fig=True)

def produce_all_figures_turn_taking():
    feature_names = npzr.get_all_features()
    for task_name in ["task5","task4A","task4B"]:
        for feature_name in feature_names:
            try:
                produce_figures_turn_taking(task_name, feature_name, n = 10000, use_density = False)
            except Exception as e:
                print(f"Failed to produce figure for {task_name} {feature_name}")
                print(traceback.format_exc())

def produce_all_figures_overall():
    feature_names = npzr.get_all_features()
    for task_name in ["task5","task4A","task4B"]:
        for feature_name in feature_names:
            try:
                produce_figures_overall(task_name, feature_name)
            except Exception as e:
                print(f"Failed to produce figure for {task_name} {feature_name}")
                print(traceback.format_exc())

def produce_all_figures():
    produce_all_figures_turn_taking()
    produce_all_figures_overall()

if __name__ == '__main__':
    globals()[sys.argv[1]]()
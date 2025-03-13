import sys
import math
import traceback
import numpy as np
import numpy_wrapper as npw
import eaf_consumer as ec
import matplotlib.pyplot as plt
import npz_reader as npzr
import io_tools as iot
import filtering as filt

def sanitize_labels(labels, tag=""):
    labels = npw.string_array(labels)
    sanitized_labels = np.empty(labels.shape, dtype = labels.dtype)
    for i, label in np.ndenumerate(labels):
        sanitized_label, sanitized_number = ec.sanitize(label)
        if sanitized_number is None:
            sanitized_number = "[all]"
        tagspace = " "
        if len(tag) > 0:
            tagspace = " " + tag + " "
        sanitized_labels[i] = f"{sanitized_number}{tagspace}{sanitized_label}"
    return sanitized_labels

def cc(data, labels, title, reorder=True, rescale_rows=True, style="discrete"):
    cmap = None
    if style == "discrete":
        cmap = "nipy_spectral"
    elif style == "uniform":
        cmap = "gist_heat"
    else:
        cmap = style
    plot_data = data
    plot_labels = labels
    if rescale_rows:
        plot_data = filt.to_01(plot_data)
    if reorder:
        plot_data, plot_labels = npzr.reorder_data(plot_data, plot_labels)
    if npw.is_string(plot_labels):
        plot_labels = [plot_labels]
    fig, (ax) = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(len(plot_labels)/4)
    ax.imshow(np.atleast_2d(plot_data), cmap=cmap, aspect='auto', interpolation='none', norm='symlog')
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=True, right=False, labelright=False)
    ax.set_title(title)
    if len(plot_labels) > 0:
        ax.set_yticks(np.arange(len(plot_labels)), labels = plot_labels)
    plt.show()

def r(data, labels, title, colorbar=False, vmin=None, vmax=None, fig = None, ax = None, labeltop = True, labelleft = True):
    individual = fig == None or ax == None
    if individual:
        fig, (ax) = plt.subplots(1, 1)
        fig.set_figwidth(15)
        fig.set_figheight(15)
    cax = ax.imshow(data, cmap="PiYG_r", aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
    if colorbar: fig.colorbar(cax)
    ax.tick_params(top=False, labeltop=labeltop, bottom=False, labelbottom=False, left=False, labelleft=labelleft, right=False, labelright=False)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha='left')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    if individual:
        plt.show()

def plot_line(feature, color, n=500, label="", smooth=False, x_axis=None):
    if np.size(feature) == 0:
        return
    if isinstance(x_axis, type(None)):
        x_axis = np.arange(feature.shape[0])
    main_alpha = 1
    if smooth: 
        feature_smooth = filt.ma(feature, {"n":n})
        feature_smooth = filt.ma(feature_smooth, {"n":n})
        feature_smooth = filt.ma(feature_smooth, {"n":n})
        plt.plot(x_axis, feature_smooth, color = color, ls='--')
        main_alpha = 0.25
    plt.plot(x_axis, feature, color = color, alpha = main_alpha, label=label)

# TODO move display utils elsewhere?
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

def plot_feature(matrix, labels, feature_name, task_name, mid_point=None, save_fig=False, force_scale=False, n=500, do_lines=True, x_is_ratio=False):
    # Data preparation
    matching_labels = npzr.has(labels, feature_name)
    matched_labels = labels[matching_labels]
    line_count = round(sum(matching_labels))
    
    unstacked_features = matrix[matching_labels,:,:]
    t = unstacked_features.shape[1]
    x_axis = np.arange(t) - 1
    if np.isscalar(mid_point):
        x_axis = x_axis - mid_point
    if x_is_ratio:
        x_axis = x_axis / np.max(np.abs(x_axis))

    features = unstacked_features.transpose(1, 0, 2).reshape(t, -1)
    features = np.nan_to_num(features)
    
    samples = round(features.shape[-1]/line_count)
    draw_smooths = samples < 30 or line_count <= 3
    draw_grid = t > 50000

    # Plots
    y_max = 0
    y_min = 0
    color = iter(plt.cm.rainbow(np.linspace(0, 1, line_count)))
    if do_lines:
        for i in range(line_count):
            label = " ".join(matched_labels[i].split(" ")[0:])
            feature = np.nanmean(features[:, i*samples:(i+1)*samples], axis=-1)
            y_max = max(y_max, np.max(feature))
            y_min = min(y_min, np.min(feature))
            plot_line(feature, color = next(color), label=f'{label}', smooth=draw_smooths, x_axis=x_axis, n=n)
    if line_count > 1 or not do_lines:
        feature_means = np.nanmean(features, axis=-1)
        y_max = max(y_max, np.max(feature_means))
        y_min = max(y_min, np.min(feature_means))
        plot_line(feature_means, color = 'black', label=f'Combined', smooth=draw_smooths, x_axis=x_axis, n=n)
    if np.isscalar(mid_point):
        plt.axvline(0, c = 'black', alpha=0.5, lw=1, label="Event Point")
    if force_scale:
        y_bot = min(0, y_min)
        y_top = max(y_max, 1)
        y_round_sym = max(round_to_closest_mult(y_bot), round_to_closest_mult(y_top))
        y_top = y_round_sym
        if y_bot < 0:
            y_bot = -y_round_sym
        plt.yticks(np.linspace(y_bot,y_top,11))
        plt.ylim(y_bot, y_top)
    if draw_grid:
        plt.grid(True, which='both', axis='y')
    plt.xlim(np.min(x_axis), np.max(x_axis))

    # Legend and Title
    title = f"Mean occurrences of \'{feature_name}\' in task \'{task_name}\'"
    modifiers = [f"(N={features.shape[-1]})"]
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
        safe_title = f"{plot_name}_{task_name}_{feature_name}"
        safe_title = safe_title.replace(":","-").replace(" ","_")
        plt.savefig(iot.figs_path() / f"{safe_title}.png")
        plt.clf()
        plt.close()
    else:
        plt.show()

def produce_figures_turn_taking(task_name, feature_name, n = 5000, use_density = False):
    total_segments = []
    # if there is no limit for ram usage, the whole data could be collected instead
    # for every segment and you would only have to do one pass of file reads
    # to produce all the graphs
    for npz in npzr.npz_list():
        if task_name in npz:
            segments = npzr.get_turn_taking_data_segments(npz, [[("has",feature_name)]], n = n, use_density = use_density)
            total_segments += segments
    TL, TD, mid_point = npzr.combined_segment_matrix_with_anchor(total_segments)
    plot_feature(TD, TL, feature_name, task_name, mid_point = mid_point, save_fig=True)
    del TL
    del TD

def produce_figures_overall(task_name, feature_name):
    total_segments = []
    # if there is no limit for ram usage, the whole data could be collected instead
    # for every segment and you would only have to do one pass of file reads
    # to produce all the graphs
    for npz in npzr.npz_list():
        if task_name in npz:
            segments = npzr.get_entire_data_as_segment(npz, [[("has",feature_name)]])
            total_segments += segments
    TL, TD = npzr.combined_segment_matrix_with_interpolation(total_segments)
    plot_feature(TD, TL, feature_name, task_name, save_fig=True, do_lines=False, x_is_ratio=True, force_scale=True, n = 20000)
    del TL
    del TD

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
import numpy as np
import eaf_consumer as ec
import matplotlib.pyplot as plt
import math
import npz_reader as npzr

def sanitize_labels(labels, tag=""):
    labels = np.asarray(labels, dtype = '<U64')
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

def display_spikes(data, labels, title, reorder=False, normrows=True, datadisplay="random"):
    colormap = None
    if datadisplay == "random":
        colormap = "nipy_spectral"
    if datadisplay == "uniform":
        colormap = "gist_heat"
    plot_data = data
    plot_labels = labels
    if normrows:
        plot_data = plot_data - np.min(plot_data, axis=1, keepdims=True)
        plot_data = plot_data / np.max(np.abs(plot_data), axis=1, keepdims=True)
    if reorder:
        plot_data, plot_labels = npzr.reorder_data(plot_data, plot_labels)
    fig, (ax) = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(len(labels)/4)
    ax.imshow(plot_data, cmap=colormap, aspect='auto', interpolation='none', norm='symlog')
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=True, right=False, labelright=False)
    ax.set_title(title)
    if len(plot_labels) > 0:
        ax.set_yticks(np.arange(len(plot_labels)), labels = plot_labels)
    plt.show()

def display_relationships(data, labels, title, colorbar=False, vmin=None, vmax=None, fig = None, ax = None, labeltop = True, labelleft = True):
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

def display_all(D, L, title, datadisplay="random"):
    display_spikes(D, L, title, reorder=True, datadisplay = datadisplay)

def display_VAD_graph(D, L, title):
    time_filter = np.any(np.expand_dims(npzr.has(L, "performance"), axis=1) & ((D == 1)), axis=0)
    VAD_filter = npzr.has(L, "vad") | npzr.has(L, "text")
    d1, l1 = npzr.select_labels(npzr.select_time(D, time_filter), L, VAD_filter)
    display_spikes(d1, l1, title + " VAD", reorder=True)

def display_overlap_graph(D, L, title):
    time_filter = np.any(np.expand_dims(npzr.has(L, "performance"), axis=1) & ((D == 1)), axis=0)
    d1, l1 = npzr.select_labels(npzr.select_time(D, time_filter), L, npzr.has(L, "text"))
    d2, l2 = npzr.select_labels(npzr.select_time(D, time_filter), L, npzr.has(L, "vad"))
    d3, l3 = npzr.select_labels(npzr.select_time(D, time_filter), L, npzr.has(L,"overlap"))
    
    overlap_data_text = np.sum(d1, axis=0, keepdims=True)
    overlap_labels_text = np.array(["+".join(l1)])
    overlap_data_vad = np.sum(d2, axis=0, keepdims=True)
    overlap_labels_vad = np.array(["+".join(l2)])
    
    overlap_data, overlap_labels = npzr.append_labels(overlap_data_text, overlap_labels_text, overlap_data_vad, overlap_labels_vad)
    overlap_data, overlap_labels = npzr.append_labels(d1, l1, overlap_data, overlap_labels)
    overlap_data, overlap_labels = npzr.append_labels(d3, l3, overlap_data, overlap_labels)
    display_spikes(overlap_data, overlap_labels, title + " overlap", reorder=True, normrows=False)

def display_energy_graph(D, L, title):
    time_filter = np.any(np.expand_dims(npzr.has(L, "performance"), axis=1) & ((D == 1)), axis=0)
    energy_filter = npzr.has(L, "energy")
    d1, l1 = npzr.select_labels(npzr.select_time(D, time_filter), L, energy_filter)
    overlap_data_energy, p = npzr.pmean(d1, p=0)
    overlap_labels_energy = np.array([f"pmean {p}"])
    overlap_data, overlap_labels = npzr.append_labels(d1, l1, overlap_data_energy, overlap_labels_energy)
    display_spikes(overlap_data, overlap_labels, title + " energy", reorder=True, normrows=False)
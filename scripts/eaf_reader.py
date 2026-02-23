import re
import numpy as np
import numpy_wrapper as npw
import naming_tools as nt
import io_tools as iot
import data_logger as dl
from pympi import Elan as elan

TEXT_TAG = "<text>"
LABELLED_TIERS = ("hand", "head", "body", "text")

def read_eaf(eafpath):
    eaf = elan.Eaf(eafpath)
    return eaf

def read_eaf_from_name(name):
    name = nt.file_swap(name, "eaf")
    return read_eaf(iot.annotation_eafs_path() / name)

def annotation_names():
    names = []
    for name in iot.list_dir_names(iot.annotation_eafs_path(), "eaf"):
        names.append(name)
    return names


def eaf_info(eaf_path):
    metadata = iot.read_metadata_from_path(eaf_path)
    return metadata["size"]


def sanitize(name, loc=False):
    sanitized_name = name
    sanitized_number = None

    # seperates speaker number
    namespace = name
    if any(x.isdigit() for x in name) and "_" in name:
        numspace = name.split("_")[0]
        numspace = numspace.replace("sspeaker", "speaker")
        namespace = "_".join(name.split("_")[1:])
        sanitized_number = numspace

    # removes hyphens and underscores, lowercases
    sanitized_name = "".join(
        filter(
            lambda x: x.isalpha() or x.isdigit() or x == ":" or x == "<" or x == ">",
            namespace,
        )
    ).lower()

    # replaces common plurals
    sanitized_name = sanitized_name.replace("pauses", "pause")
    sanitized_name = sanitized_name.replace("overlaps", "overlap")
    sanitized_name = sanitized_name.replace("hands", "hand")
    if loc and name != sanitized_name:
        dl.write_to_manifest_log(dl.CORRECTION_TYPE, f"Correction: {name} => {sanitized_name} @ {loc}")

    return sanitized_name, sanitized_number

def sanitize_label(label, loc=False):
    sanitized_label = label.lower()
    sanitized_label = sanitized_label.replace("<", "")
    sanitized_label = sanitized_label.replace(">", "")
    sanitized_label = sanitized_label.strip()
    if " " in sanitized_label:
        sanitized_label = "other"
    if "other" in sanitized_label:
        sanitized_label = "other"
    if "nod" in sanitized_label:
        sanitized_label = "nodding"
    if len(sanitized_label) <= 1:
        sanitized_label = "other"
    if "pauses" in sanitized_label:
        sanitized_label = "pause"
    if "overlaps" in sanitized_label:
        sanitized_label = "overlap"
    if "hands" in sanitized_label:
        sanitized_label = "hand"
    if loc and label != sanitized_label:
        dl.write_to_manifest_log(dl.CORRECTION_TYPE, f"Correction: {label} => {sanitized_label} @ {loc}")
    return sanitized_label

def section_text(text):
    tp = 0
    tl = len(text)
    backchannel_regex = r"<.*?>"
    sections = []
    for m in re.finditer(backchannel_regex, text):
        span = m.span()
        wc = len(text[tp:span[0]].strip().split(" "))
        sections.append((TEXT_TAG, span[0] - tp, wc))
        sections.append((m[0], span[1] - span[0], 1))
        tp = span[1]
    wc = len(text[tp:tl].strip().split(" "))
    sections.append((TEXT_TAG, tl - tp, wc))
    return sections


def find_text_tokens(text, textual_tokens="ignore", nontextual_tokens="ignore"):
    if textual_tokens == "ignore" and nontextual_tokens == "ignore":
        dl.log(
            "Ignoring both textual and non-textual text! Is this what you wanted to do?"
        )
        return ""

    sections = section_text(text)
    counter = dict()
    for tag, char_length, word_length in sections:
        tokenizing_behaviour = "undefined"
        if tag == TEXT_TAG:
            tokenizing_behaviour = textual_tokens
        if tag != TEXT_TAG:
            tokenizing_behaviour = nontextual_tokens

        if tokenizing_behaviour == "ignore":
            continue

        if tag not in counter:
            counter[tag] = False
        if tokenizing_behaviour == "contains":
            counter[tag] = True
        if tokenizing_behaviour == "count":
            counter[tag] += 1
        if tokenizing_behaviour == "sum_characters":
            counter[tag] += char_length
        if tokenizing_behaviour == "sum_words":
            counter[tag] += word_length


    tokens = dict()
    for tag in sorted(counter):
        count = counter[tag]
        if isinstance(count, bool):
            if not count:
                continue
        elif count < 1:
            continue
        tokens[tag] = counter[tag]
    return tokens

BACKCHANNEL_TOKEN = "<bc>"
TRANSLATION_TOKEN = "<trans>"
GARBAGE_TOKEN = "<garbage>"
LAUGHING_TOKEN = "<laughing>"
PARAL_TOKEN = "<paral>"
HESITATION_TOKEN = "<hesitation>"

def sanitize_text(text, collapse_languages=True, loc=False):
    sanitized_text = text.lower().strip()

    substitution_map = {
        BACKCHANNEL_TOKEN: ["<bacch>", "<backh>", "<bakch>", "<backch>", "<bachch>"],
        GARBAGE_TOKEN: [
            "<gabage>",
            "<garbage>",
            "<gargbage>",
            "<garbege>",
            "<unk>",
            "<other>",
            "<bgnoise>",
        ],
        LAUGHING_TOKEN: ["<laugh>", "<laughin>", "<laught>"],
        HESITATION_TOKEN: [
            "<hesitaiton>",
            "<hestitation>",
            "<hesiation>",
            "<heistation>",
            "<hesitiation>",
            "<hesitaion>",
            "<hesittaion>",
            "<hesitationhesitation>"
        ],
        PARAL_TOKEN: ["<para>"],
    }
    if collapse_languages:
        substitution_map[TRANSLATION_TOKEN] = ["<transen>", "<transjp>", "<transfr>", "<transsv>"]

    for substitution in substitution_map:
        for sequence in substitution_map[substitution]:
            if sequence not in sanitized_text:
                continue
            if loc: 
                dl.write_to_manifest_log(dl.CORRECTION_TYPE, f"Correction: {sequence} => {substitution} @ {loc}")
            sanitized_text = sanitized_text.replace(sequence, substitution)

    return sanitized_text


def get_labels_for_tier(tier, tier_annotations, loc=False):
    labels = dict()
    for t0, t1, string in tier_annotations:
        tokens_values = {string: True}
        if tier == "text":
            loc_t = False
            if loc:
                loc_t = f"{loc}[{tier}:{t0}-{t1}]"
            tokens_values = find_text_tokens(
                sanitize_text(string, loc=loc_t), textual_tokens="sum_words", nontextual_tokens="sum_words"
            )

        for token_key in tokens_values:
            label = token_key
            value = tokens_values[token_key]
            if isinstance(value, bool):
                value = int(value)
            loc_t = False
            if loc:
                loc_t = f"{loc}[{tier}:{t0}-{t1}]"
            label = sanitize_label(label, loc=loc_t)
            if label not in labels:
                labels[label] = []
            labels[label].append((t0, t1, string, value))
    return labels


def add_annotation_to_timeseries(a, annotation):
    annotation_start = annotation[0]
    annotation_end = annotation[1]
    value = 1
    if len(annotation) >= 4:
        value = annotation[3]
    a[annotation_start:annotation_end] = value


def add_annotations_to_timeseries(a, annotations):
    for annotation in annotations:
        add_annotation_to_timeseries(a, annotation)


BLOCK_LIST = [f"{nt.TEXT_TYPE}:other", f"{nt.TEXT_TYPE}:hesitationhesitation", f"{nt.TEXT_TYPE}:other", f"{nt.TEXT_TYPE}:bgnoise"]


def block_listed(key, name="unknown eaf"):
    for block in BLOCK_LIST:
        if block in key:
            dl.log(
                f"Ignoring label '{block}' while processing '{name}' based on block-list."
            )
            return True
    return False

def eaf_to_data_matrix(eaf, width=None, name="unknown eaf"):
    if isinstance(width, type(None)):
        width = eaf.get_full_time_interval()[1]

    # annotate timeseries
    tiers = list(eaf.tiers)

    # get annotations
    annotations = dict()
    for tier in tiers:
        annotations[tier] = eaf.get_annotation_data_for_tier(tier)

    annotation_series = dict()
    for tier in tiers:
        sanitized_tier, sanitized_number = sanitize(tier, loc=name)

        if not isinstance(sanitized_number, type(None)):
            tier_name = f"{sanitized_number}_{sanitized_tier}"
        else:
            tier_name = sanitized_tier

        if sanitized_tier in LABELLED_TIERS:
            label_annotations = get_labels_for_tier(sanitized_tier, annotations[tier], loc=name)

            for label in label_annotations:
                series = np.zeros(width)
                add_annotations_to_timeseries(series, label_annotations[label])
                key = f"{tier_name}:{label}"
                if block_listed(key, name):
                    continue
                if key not in annotation_series:
                    annotation_series[key] = np.zeros(width)
                annotation_series[key] = annotation_series[key] + series
        
        if sanitized_tier not in LABELLED_TIERS:
            series = np.zeros(width)
            key = f"{tier_name}"
            if block_listed(key, name):
                continue
            if key not in annotation_series:
                annotation_series[key] = np.zeros(width)
            add_annotations_to_timeseries(series, annotations[tier])
            annotation_series[key] = annotation_series[key] + series

    # create data matrix
    data = np.zeros((width, len(annotation_series)))
    labels = []
    for i, label in enumerate(annotation_series):
        data[:, i] = annotation_series[label]
        feature, source = sanitize(label)
        if isinstance(source, type(None)):
            source = nt.SPEAKERS
        # repair for unknown labels, assumes shared label
        if source == nt.SPEAKERNONE:
            source = nt.SPEAKERS
        # repair for untyped labels, assumes IC
        if ":" not in feature:
            feature = f"{nt.IC_TYPE}:{feature}"
        labels.append(nt.create_label(source, nt.ANNOTATION_TAG, feature))

    labels = npw.string_array(labels)
    return data, labels

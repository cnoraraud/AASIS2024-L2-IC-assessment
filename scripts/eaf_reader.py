import re
import numpy as np
import numpy_wrapper as npw
import naming_tools as nt
import io_tools as iot
import data_logger as dl

TEXT_TAG = "<text>"
LABELLED_TIERS = ("hand", "head", "body", "text")


def annotation_names():
    names = []
    for name in iot.list_dir_names(iot.annotation_eafs_path(), "eaf"):
        names.append(name)
    return names


def eaf_info(eaf_path):
    metadata = iot.read_metadata_from_path(eaf_path)
    return metadata["size"]


def sanitize(name):
    sanitized_name = name
    sanitized_number = None

    # seperates speaker number
    namespace = name
    if any(x.isdigit() for x in name) and "_" in name:
        numspace = name.split("_")[0]
        numspace = numspace.replace("sspeaker", "speaker")
        namespace = "_".join(name.split("_")[1:])
        sanitized_number = numspace

    # removes hyphens and underscores
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

    return sanitized_name, sanitized_number


def section_text(text):
    tp = 0
    tl = len(text)
    backchannel_regex = r"<.*?>"
    sections = []
    for m in re.finditer(backchannel_regex, text):
        span = m.span()
        sections.append((TEXT_TAG, span[0] - tp))
        sections.append((m[0], span[1] - span[0]))
        tp = span[1]
    sections.append((TEXT_TAG, tl - tp))
    return sections


def find_text_tokens(text, textual_tokens="ignore", nontextual_tokens="ignore"):
    if textual_tokens == "ignore" and nontextual_tokens == "ignore":
        dl.log(
            "Ignoring both textual and non-textual text! Is this what you wanted to do?"
        )
        return ""

    sections = section_text(text)
    counter = dict()
    for tag, length in sections:
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
        if tokenizing_behaviour == "sum":
            counter[tag] += length

    tokens = []
    for tag in sorted(counter):
        count = counter[tag]
        if isinstance(count, bool):
            if count:
                tokens.append(f"{tag}")
        elif count >= 1:
            tokens.append(f"{tag}:{count}")
    return "+".join(tokens)


def sanitize_label(label):
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
    return sanitized_label


def sanitize_text(text, collapse_languages=True):
    sanitized_text = text.lower().strip()

    if collapse_languages:
        sanitized_text = sanitized_text.replace("<transen>", "<trans>")
        sanitized_text = sanitized_text.replace("<transjp>", "<trans>")
        sanitized_text = sanitized_text.replace("<transfr>", "<trans>")

    substitution_map = {
        "<trans>": ["<transen>", "<transjp>", "<transfr>", "<transsv>"],
        "<bc>": ["<bacch>", "<backh>", "<bakch>", "<backch>", "<bachch>"],
        "<garbage>": [
            "<gabage>",
            "<garbage>",
            "<gargbage>",
            "<garbege>",
            "<unk>",
            "<other>" "<bgnoise>",
        ],
        "<laughing>": ["<laugh>", "<laughin>", "<laught>"],
        "<hesitation>": [
            "<hesitaiton>",
            "<hestitation>",
            "<hesiation>",
            "<heistation>",
            "<hesitiation>",
            "<hesitaion>",
            "<hesittaion>",
        ],
        "<paral>": ["<para>"],
    }

    for substitution in substitution_map:
        if substitution == "<trans>" and not collapse_languages:
            continue
        sequences = substitution_map[substitution]
        for sequence in sequences:
            sanitized_text = sanitized_text.replace(sequence, substitution)

    return sanitized_text


def get_labels_for_tier(tier, tier_annotations):
    labels = dict()
    for t0, t1, string in tier_annotations:
        tokens_values_string = string
        if tier == "text":
            tokens_values_string = find_text_tokens(
                sanitize_text(string), textual_tokens="sum", nontextual_tokens="count"
            )

        # This is dumb, I should pass a data structure
        tokens_values = tokens_values_string.split("+")
        for token_value in tokens_values:
            label = "undefined"
            value = 1
            if ":" in token_value:
                label = token_value.split(":")[0]
                value = int(token_value.split(":")[1])
            else:
                label = token_value
            label = sanitize_label(label)
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


BLOCK_LIST = ["text:other", "text:hesitationhesitation", "hand:other", "text:bgnoise"]


def block_listed(key, name="unknown eaf"):
    for block in BLOCK_LIST:
        if block in key:
            dl.log(
                f"Ignoring label '{block}' while processing '{name}' based on block-list."
            )
            return True
    return False


def sanitize_labels(labels, tag=""):
    labels = npw.string_array(labels)
    sanitized_labels = np.empty(labels.shape, dtype=labels.dtype)
    for i, label in np.ndenumerate(labels):
        sanitized_label, sanitized_number = sanitize(label)
        if sanitized_number is None:
            sanitized_number = "[all]"
        tagspace = " "
        if len(tag) > 0:
            tagspace = " " + tag + " "
        sanitized_labels[i] = f"{sanitized_number}{tagspace}{sanitized_label}"
    return sanitized_labels


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
        sanitized_tier, sanitized_number = sanitize(tier)

        if not isinstance(sanitized_number, type(None)):
            tier_name = f"{sanitized_number}_{sanitized_tier}"
        else:
            tier_name = sanitized_tier

        if sanitized_tier in LABELLED_TIERS:
            label_annotations = get_labels_for_tier(sanitized_tier, annotations[tier])

            for label in label_annotations:
                series = np.zeros(width)
                add_annotations_to_timeseries(series, label_annotations[label])
                key = f"{tier_name}:{label}"
                if block_listed(key, name):
                    continue
                if key not in annotation_series:
                    annotation_series[key] = np.zeros(width)
                annotation_series[key] = annotation_series[key] + series
        else:
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
            source = nt.ALL_SOURCE
        labels.append(nt.create_label(source, nt.ANNOTATION_TAG, feature))

    labels = npw.string_array(labels)
    return data, labels

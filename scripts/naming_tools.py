import re
import pathlib as p
from collections import Counter

INIT_L_KEY = "initiator_labels"
RESP_L_KEY = "responder_labels"
SELF_L_KEY = "self_labels"
OTHR_L_KEY = "other_labels"

SPEAKER_SELF = "self"
SPEAKER_OTHER = "other"
SPEAKER1 = "S001"
SPEAKER2 = "S002"
SPEAKERS = "[all]"
SPEAKERNONE = "[none]"

AA_TAG = "ic"
ANNOTATION_TAG = "(ann.)"
EXTRACTION_TAG = "(ext.)"
UNKNOWN_TAG = "(unk.)"

COLLAPSE_NONE = "none"
COLLAPSE_INITIATOR_LABELS = "initiator"
COLLAPSE_RESPONDER_LABELS = "reactive"

PROACTIVE = "proactive"
REACTIVE = "reactive"
NORMALIZABLE = "normalizable"
GENERAL = "general"

PROACTIVE_FEATURES_LIST = sorted(('mean_segment_density', 'mean_segment_width', 'segment_count', 'std_segment_density', 'std_segment_width', 'total_segment_mass', 'total_segment_width', 'var_segment_density', 'var_segment_width'))
REACTIVE_FEATURES_LIST = sorted(('mean_difference', 'mean_times_relative', 'mean_times_from_start', 'mean_times_from_end', 'median_difference', 'median_times_relative', 'median_times_from_start', 'median_times_from_end', 'std_overlap', 'std_delay', 'std_segment_overlap_count', 'std_segment_delay_count', 'std_segment_overlap_ratio', 'std_difference', 'std_times_relative', 'std_times_from_start', 'std_times_from_end', 'count_delay', 'count_overlap', 'mean_delay', 'mean_overlap', 'mean_segment_delay_count', 'mean_segment_overlap_count', 'mean_segment_overlap_ratio', 'median_delay', 'median_overlap', 'median_segment_delay_count', 'median_segment_overlap_count', 'median_segment_overlap_ratio', 'pearson_corr', 'spearman_corr', 'total_overlap'))
NORMALIZABLE_FEATURES_LIST = sorted(('segment_count', 'total_segment_mass', 'total_segment_width', 'count_delay', 'count_overlap', 'total_overlap'))

COLLAPSE_INITIATOR_LIST = sorted(PROACTIVE_FEATURES_LIST)
COLLAPSE_RESPONDER_LIST = sorted(REACTIVE_FEATURES_LIST)

ALL_FEATURES_LIST = sorted(PROACTIVE_FEATURES_LIST + REACTIVE_FEATURES_LIST)

def get_feature_type(feature):
    normalizable = ""
    feature_type = ""
    if feature in NORMALIZABLE_FEATURES_LIST:
        normalizable = NORMALIZABLE
    if feature in PROACTIVE_FEATURES_LIST:
        feature_type = PROACTIVE
    elif feature in REACTIVE_FEATURES_LIST:
        feature_type = REACTIVE 
    return feature_type, normalizable

def is_speaker_related(speaker, value):
    check_substring = value.split(" ")[0]
    if speaker in check_substring:
        return True
    if SPEAKERS in check_substring:
        return True
    return False

def switch_label_speaker(label, speaker):
    if " " not in label:
        return label
    return f"{speaker} {" ".join(label.split(" ")[1:])}"

def get_S_label(label, reference_S):
    reference_role = get_session_role(label)
    label_S = get_speaker(reference_S, reference_role)
    return switch_label_speaker(label, label_S)

def get_session_role(label):
    if " " not in label:
        return None
    candidate = label.split(" ")[0]
    if SPEAKER_SELF == candidate:
        return SPEAKER_SELF
    if SPEAKER_OTHER == candidate:
        return SPEAKER_OTHER
    if SPEAKERS == candidate:
        return SPEAKERS
    return SPEAKERNONE

def get_speakers():
    return [SPEAKER1, SPEAKER2]

def get_speakers_all():
    return [SPEAKER1, SPEAKER2, SPEAKERS]

def get_speaker(speaker, role):
    if role == SPEAKER_SELF:
        return speaker
    if role == SPEAKER_OTHER:
        return get_speaker_other(speaker)
    if role == SPEAKERS:
        return SPEAKERS
    return SPEAKERNONE

def get_speaker_other(speaker):
    if speaker == SPEAKER1:
        return SPEAKER2
    if speaker == SPEAKER2:
        return SPEAKER1
    # immutable
    if speaker == SPEAKERS:
        return SPEAKERS
    return SPEAKERNONE

def get_name(path):
    return p.Path(path).name

def file_swap(path, to=None, all=True):
    if not all:
        path = get_name(path)
    if isinstance(to, type(None)):
        return p.Path(path).stem
    else:
        to = to.replace(".","")
        return f"{p.Path(path).stem}.{to}"

def find_speakers(name):
    name = get_name(name)
    r_speaker = r"speaker\d+"
    speakers = []
    for speaker_match in re.finditer(r_speaker,name):
        speaker = speaker_match[0]
        speakers.append(speaker)
    return speakers

def find_task(name):
    name = get_name(name)
    r_task = r"task\d[a-zA-Z]*"
    task = None
    for task_match in re.finditer(r_task,name):
        task = task_match[0]
    return task

def find_cams(name):
    name = get_name(name)
    r_cam = r"cam(e(r(a?)?)?)?\d+"
    cams = []
    for cam_match in re.finditer(r_cam,name):
        cam = cam_match[0]
        cams.append(cam)
    return cams

def find_mics(name):
    name = get_name(name)
    r_mic = r"mic(r(o(p(h(o(n(e?)?)?)?)?)?)?)?\d+"
    mics = []
    for mic_match in re.finditer(r_mic,name):
        mic = mic_match[0]
        mics.append(mic)
    return mics

def get_at_most_nth(slots, n=3):
    candidate = None
    if len(slots) >= n:
        candidate = slots[n-1]
    elif len(slots) > 0:
        candidate = slots[-1]
    return candidate

def get_last_slots_as_candidates(slots, n=2):
    candidates = []
    for i in range(1,n+1):
        if len(slots) >= i:
            candidates.append(slots[-i])
    return candidates

def find_annotator(name):
    name = file_swap(get_name(name))
    slots = name.split("_")
    suitable = "unknown_annotator"
    for candidate in get_last_slots_as_candidates(slots, 2):
        r_ann = r"[a-zA-Z]{2}"
        if not re.match(r_ann, candidate):
            continue
        suitable = candidate.lower()
        break
    return suitable

def find_version(name):
    name = file_swap(get_name(name))
    slots = name.split("_")
    suitable = "unknown_version"

    r_version = r"(^|-|take|v(er(s(ion)?)?)?)\d+"
    r_number = r"\d+"
    for candidate in get_last_slots_as_candidates(slots, 2):
        version_numbers = []
        for version_match in re.finditer(r_version, candidate):
            for number_match in re.finditer(r_number, version_match[0]):
                match_val = number_match[0]
                # Probably unnecessary
                if match_val.isnumeric():
                    version_numbers.append(to_name_number(match_val))
                
        if len(version_numbers) > 0:
            suitable = "-".join(version_numbers)
            break
    return suitable

def find_channel(label):
    slots = label.split(" ")
    candidate = get_at_most_nth(slots, 3)
    if candidate:
        return candidate
    return "unknown_feature"

def find_tag(label):
    if ANNOTATION_TAG in label:
        return ANNOTATION_TAG
    if EXTRACTION_TAG in label:
        return EXTRACTION_TAG
    return 

def find_sources(label):
    slots = label.split(" ")
    if len(slots) == 0:
        return "unknown_source"
    candidate = slots[0]
    candidate = candidate.split(":")[0]
    sources = []
    if SPEAKERS in candidate:
        sources.append(SPEAKERS)
    if SPEAKER1 in candidate:
        sources.append(SPEAKER1)
    if SPEAKER2 in candidate:
        sources.append(SPEAKER2)
    sources.extend(to_sources(candidate))
    return sources

def to_name_number(number_string):
    number_int = int(number_string)
    number = str(number_int)
    number = number.rjust(3, "0")
    return number

def to_sources(candidate):
    sources = []
    r_number = r"\d+"
    for number_match in re.finditer(r_number, candidate):
        sources.append(to_name_number(number_match[0]))
    return sources

def speakers_to_sources(speakers):
    sources = []
    for speaker in speakers:
        sources.extend(find_sources(speaker))
    return sources

def task_names_no_prefix(tasks):
    new_tasks = []
    
    for task in tasks:
        new_task = str(task).lower()
        new_task = new_task.replace("task","")
        new_tasks.append(new_task)

    return new_tasks

def task_names_with_prefix(tasks):
    new_tasks = []
    
    for task in tasks:
        new_task = task.lower()
        if "task" not in new_task:
            new_task = f"task{new_task}"
        new_tasks.append(new_task)

    return new_tasks

def get_anon_source(source):
    if source == SPEAKERS:
        return source
    if source == SPEAKERNONE:
        return source
    if source.isnumeric():
        if int(source) % 2 != 0:
            return SPEAKER1
        if int(source) % 2 == 0:
            return SPEAKER2
    return SPEAKERNONE

def find_best_candidate(candidates):
    c = Counter(candidates)
    best = []
    best_count = 0
    for key in c:
        count = c[key]
        if count > best_count:
            best.clear()
            best_count = count
        if count == best_count:
            best.append(key)
    return best

def compact_sources(sources, plural_nicks = False):
    if len(sources) == 1:
        return sources[0]
    if plural_nicks and len(sources) == 0:
        return SPEAKERNONE
    if plural_nicks and len(sources) > 1:
        return SPEAKERS
    if len(sources) == 0:
        return ""
    return "-".join(sorted(list(set(sources))))

def find_extra(label):
    slots = label.split(" ")
    if len(slots) < 4:
        return None
    return " ".join(slots[3:])

def create_label(source, tag, feature, extra=None):
    base_label = f"{source} {tag} {feature}"
    if isinstance(extra, type(None)):
        return base_label
    return f"{base_label} {extra}"

def sanitize_filename(filename):
    filename = filename.replace(":","-")
    filename = filename.replace(" ","_")
    filename = filename.replace(" \"", "[")
    filename = filename.replace("\" ", "]")
    filename = filename.replace("\"", "#")
    filename = filename.replace(".", "_")
    filename = filename.replace("|", "_")
    filename = filename.replace("?", "")
    filename = filename.replace("*","")
    filename = filename.replace("/","_")
    filename = filename.replace("\\","_")
    filename = filename.replace("<","(")
    filename = filename.replace(">",")")
    filename = filename.replace("&","and")
    return filename
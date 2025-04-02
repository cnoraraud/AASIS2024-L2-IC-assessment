import re
import pathlib as p
from collections import Counter
import numpy_wrapper as npw

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

def find_feature(label):
    slots = label.split(" ")
    candidate = get_at_most_nth(slots, 3)
    if candidate: return candidate
    return "unknown_feature"

AA_TAG = "ic"
ANNOTATION_TAG = "(ann.)"
EXTRACTION_TAG = "(ext.)"
UNKNOWN_TAG = "(unk.)"
def find_tag(label):
    if ANNOTATION_TAG in label:
        return ANNOTATION_TAG
    if EXTRACTION_TAG in label:
        return EXTRACTION_TAG
    return 

ALL_SOURCE = npw.SPEAKERS
def find_sources(label):
    slots = label.split(" ")
    if len(slots) == 0:
        return "unknown_source"
    candidate = slots[0]
    candidate = candidate.split(":")[0]
    sources = []
    if ALL_SOURCE in candidate:
        sources.append(ALL_SOURCE)
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

def get_anon_source(source):
    if source == npw.SPEAKERS:
        return source
    if source == npw.SPEAKERNONE:
        return source
    if source.isnumeric():
        if int(source) % 2 != 0:
            return npw.SPEAKER1
        if int(source) % 2 == 0:
            return npw.SPEAKER2
    return npw.SPEAKERNONE

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
    if len(sources) == 1: return sources[0]
    if plural_nicks and len(sources) == 0: return npw.SPEAKERNONE
    if plural_nicks and len(sources) > 1: return npw.SPEAKERS
    if len(sources) == 0: return ""
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
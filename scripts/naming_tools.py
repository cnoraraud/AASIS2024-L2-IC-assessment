import re
import pathlib as p
import numpy_wrapper as npw

def get_name(path):
    return p.Path(path).name

def file_swap(path, to=None):
    to = to.replace(".","")
    if isinstance(to, type(None)):
        return p.Path(path).stem
    else:
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

def find_cam(name):
    name = get_name(name)
    r_cam = r"cam(e(r(a?)?)?)?\d+"
    cams = []
    for cam_match in re.finditer(r_cam,name):
        cam = cam_match[0]
        cams.append(cam)
    return cams

def find_mic(name):
    name = get_name(name)
    r_mic = r"mic(r(o(p(h(o(n(e?)?)?)?)?)?)?)?\d+"
    mics = []
    for mic_match in re.finditer(r_mic,name):
        mic = mic_match[0]
        mics.append(mic)

def find_annotator(name):
    name = file_swap(get_name(name))
    slots = name.split("_")
    if len(slots) < 1:
        return None
    candidate = slots[-1]
    r_ann = r"\w\w"
    if not re.match(r_ann, candidate):
        return None
    return candidate.lower()

def find_feature(label):
    slots = label.split(" ")
    if len(slots) == 0:
        return "unknown_feature"
    elif len(slots) < 3:
        return slots[-1]
    else:
        return slots[2]

ANNOTATION_TAG = "(ann.)"
EXTRACTION_TAG = "(ext.)"
UNKNOWN_TAG = "(unk.)"
def find_tag(label):
    if ANNOTATION_TAG in label:
        return ANNOTATION_TAG
    if EXTRACTION_TAG in label:
        return EXTRACTION_TAG
    return 

ALL_SOURCE = "[all]"
def find_sources(label):
    slots = label.split(" ")
    if len(slots) == 0:
        return "unknown source"
    candidate = slots[0]
    candidate = candidate.split(":")[0]
    sources = []
    if ALL_SOURCE in candidate:
        sources.append(ALL_SOURCE)
    sources.extend(to_sources(candidate))
    return sources

def to_sources(candidate):
    sources = []
    r_number = r"\d+"
    for number_match in re.finditer(r_number, candidate):
        number = number_match[0]
        number_int = int(number)
        number = str(number_int)
        number = number.rjust(3, "0")
        sources.append(number)
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
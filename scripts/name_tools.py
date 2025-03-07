import re

def find_speakers(name):
    r_speaker = r"speaker\d+"
    speakers = []
    for speaker_match in re.finditer(r_speaker,name):
        speaker = speaker_match[0]
        speakers.append(speaker)
    return speakers

def find_task(name):
    r_task = r"task\d[a-zA-Z]*"
    task = None
    for task_match in re.finditer(r_task,name):
        task = task_match[0]
    return task
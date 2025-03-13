import itertools
import numpy as np
import numpy_wrapper as npw
import collections
import pandas as pd
import math
from matplotlib import pyplot as plt

def get_speaker_groups(speakers, columns):
    if isinstance(columns, str):
        columns = [columns]
    values = []
    for column in columns:
        values.append(speakers[column].dropna().unique())
    groups = dict()
    existant = 0
    nonexistant = 0
    nonexistants = []
    for grouping in itertools.product(*values):
        name_components = []
        cond = None
        for column, group in zip(columns, grouping):
            name_components.append(f"{group}")
            new_cond = speakers[column] == group
            if cond is None:
                cond = new_cond
            else:
                cond = cond & new_cond
        speakers_in_group = speakers[cond]
        name = "/".join(name_components)
        if len(speakers_in_group) > 0:
            speaker_ids = speakers_in_group["SpeakerID"].unique()
            groups[name] = speaker_ids
            existant +=1
        else:
            nonexistants.append(name)
            nonexistant +=1
    print(f"Out of {existant + nonexistant} groupings {existant} existed and {nonexistant} didn't.")
    if nonexistant > 0:
        print(f"The groups that didn't exist were: \n\t{",".join(nonexistants)}")
    return groups

def find_sample_groupins(groups, samples):
    samples_obj = dict()
    for group_key in groups:
        ids = groups[group_key]
        group_samples_obj = dict()
        for speaker in ids:
            speaker_samples = samples[(samples["speaker_1"] == speaker) | (samples["speaker_2"] == speaker)]
            speaker_samples_obj = {}
            for i, speaker_sample in speaker_samples.iterrows():
                S = None
                if speaker_sample["speaker_1"] == speaker:
                    S = npw.SPEAKER1
                if speaker_sample["speaker_2"] == speaker:
                    S = npw.SPEAKER2
                speaker_samples_obj = {
                    "speaker": speaker,
                    "S": S,
                    "npz": speaker_sample["npz"],
                    "npy": speaker_sample["npy"],
                    "task": speaker_sample["task"],
                    "annotator": speaker_sample["annotator"]}
            group_samples_obj[speaker] = speaker_samples_obj
        samples_obj[group_key] = group_samples_obj
    return samples_obj

def get_grouping_columns():
    grouping_columns = [
        "University","Gender","Speaking",
        "Interaction","Speaking vs Interaction",
        "Interlocutor Familiarity","Age Group",
        "Topic Affinity","Skill Representivity",
        "Learning", "1st Language Count",
        "2nd Language Count", "Language Count",
        "Interlocutor First Language Familiarity",
        "Gender Match","Speaking Difference",
        "Interaction Difference"
    ]
    return grouping_columns

def language_counts(speakers, key="Second Languages"):
    lc = collections.Counter()
    for ls in list(speakers[key]):
        if not isinstance(ls,str):
            continue
        for l in ls.split("/"):
            lc[l] += 1
    return lc

def get_dummy(df, default="Same", dtype="str"):
    if dtype == "str":
        return np.full(len(df), default, dtype='<U13')
    else:
        return np.zeros(len(df))

def find_language_counts(speakers):
    first_language_counts = get_dummy(speakers, 0, dtype="int")
    second_language_counts = get_dummy(speakers, 0, dtype="int")
    total_language_counts = get_dummy(speakers, 0, dtype="int")
    for i, speaker_languages in enumerate(zip(speakers["First Languages"], speakers["Second Languages"])):
        first_languages = speaker_languages[0]
        second_languages = speaker_languages[1]
        count = 0
        second_languages_set = set()
        first_languages_set = set()
        if isinstance(second_languages, str):
            second_languages_set.update(second_languages.split("/"))
        if isinstance(first_languages, str):
            first_languages_set.update(first_languages.split("/"))
        total_languages_set = first_languages_set | second_languages_set
        first_language_counts[i] = len(first_languages_set)
        second_language_counts[i] = len(second_languages_set)
        total_language_counts[i] = len(total_languages_set)
    return first_language_counts, second_language_counts, total_language_counts

def add_language_difference(speakers):
    language_pref = get_dummy(speakers, 0, dtype="int")
    language_pref[np.array(speakers["Speaking"] < speakers["Interaction"])] = -1
    language_pref[np.array(speakers["Speaking"] > speakers["Interaction"])] = 1
    language_pref[np.array(speakers["Speaking"] == speakers["Interaction"])] = 0
    speakers["Speaking vs Interaction"] = language_pref

def add_language_counts(speakers):
    first_language_counts, second_language_counts, total_language_counts = find_language_counts(speakers)
    speakers["1st Language Count"] = first_language_counts
    speakers["2nd Language Count"] = second_language_counts
    speakers["Language Count"] = total_language_counts

def add_partner_id(speakers, samples):
    base = samples[["speaker_1","speaker_2"]].drop_duplicates()
    a = base.rename(columns={"speaker_1":"SpeakerID","speaker_2":"OtherSpeakerID"})
    b = base.rename(columns={"speaker_2":"SpeakerID","speaker_1":"OtherSpeakerID"})
    other_speakers = pd.concat([a,b])
    speaker_map = dict(zip(other_speakers["SpeakerID"], other_speakers["OtherSpeakerID"]))
    other_speakers = []
    for speaker_id in speakers["SpeakerID"]:
        other_id = None
        if speaker_id in speaker_map:
            other_id = round(speaker_map[speaker_id])
        other_speakers.append(other_id)
    speakers["OtherSpeakerID"] = other_speakers

def get_languages(ls):
    if not isinstance(ls,str):
        return []
    else:
        return ls.split("/")

def add_partner_relation(speakers):
    speakers_paired = speakers.merge(speakers, how="left", left_on="OtherSpeakerID", right_on="SpeakerID")
    # language addition
    first_language_overlaps = []
    for i, speaker_pair in speakers_paired.iterrows():
        l1 = get_languages(speaker_pair["First Languages_x"])
        l2 = get_languages(speaker_pair["First Languages_y"])
        language_overlaps = int(len(set(l1) & set(l2)) > 0)
        first_language_overlaps.append(language_overlaps)
    speakers["Interlocutor First Language Familiarity"] = first_language_overlaps
    # gender match
    gender_match = get_dummy(speakers, 0, dtype="int")
    gender_match[np.array(speakers_paired["Gender_x"] == speakers_paired["Gender_y"])] = 1
    gender_match[np.array(speakers_paired["Gender_x"] != speakers_paired["Gender_y"])] = 0
    speakers["Gender Match"] = gender_match
    # speaking difference
    speaking_match = get_dummy(speakers, 0, dtype="int")
    speaking_match[np.array(speakers_paired["Speaking_x"] < speakers_paired["Speaking_y"])] = -1
    speaking_match[np.array(speakers_paired["Speaking_x"] > speakers_paired["Speaking_y"])] = 1
    speaking_match[np.array(speakers_paired["Speaking_x"] == speakers_paired["Speaking_y"])] = 0
    speakers["Speaking Difference"] = speaking_match
    # interaction difference
    interaction_match = get_dummy(speakers, 0, dtype="int")
    interaction_match[np.array(speakers_paired["Interaction_x"] < speakers_paired["Interaction_y"])] = -1
    interaction_match[np.array(speakers_paired["Interaction_x"] > speakers_paired["Interaction_y"])] = 1
    interaction_match[np.array(speakers_paired["Interaction_x"] == speakers_paired["Interaction_y"])] = 0
    speakers["Interaction Difference"] = speaking_match

def plot_column_groups(df, columns, cols = 5):
    items = len(columns)
    cols = 5
    rows = math.ceil(items/cols)
    plt.figure(figsize=(15,3*rows))
    plt.tight_layout()
    for i, col in enumerate(columns):
        keyword = col
        plt.subplot(rows, cols, i+1)
        keys = sorted(list(df[keyword].value_counts().keys()))
        values = list(df[keyword].value_counts()[keys])
        plt.pie(values, labels=keys, wedgeprops={"width":0.5})
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.title(keyword)
    plt.show()

def merge_all(speakers, samples):
    a = samples.drop(columns=["speaker_1"]).rename(columns={"speaker_2":"SpeakerID"})
    b = samples.drop(columns=["speaker_2"]).rename(columns={"speaker_1":"SpeakerID"})
    sessions_concat = pd.concat([a,b])
    sessions_concat = sessions_concat.reset_index()
    merged_table = speakers.merge(sessions_concat,how="outer",left_on="SpeakerID",right_on="SpeakerID")
    return merged_table
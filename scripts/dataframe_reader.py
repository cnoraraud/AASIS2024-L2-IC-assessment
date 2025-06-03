import math
import itertools
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy_wrapper as npw

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
        "Interaction Difference", "Score", "Score2",
        "Combined IC CEFR"
    ]
    return grouping_columns

def language_counts(speakers, key="Second Languages"):
    lc = Counter()
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

def num_to_cefr(series):
    group_map = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}
    return series.round(0).astype('Int64').map(group_map)

def cefr_to_num(series):
    group_map = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    return series.map(group_map).round(0).astype('Int64')

def add_ic_score(speakers):
    group_map = {True: "Top Half", False: "Bottom Half"}
    score = (cefr_to_num(speakers["Speaking"]) + cefr_to_num(speakers["Interaction"])) / 2.0
    speakers["self_ic_calc"] = score
    score_group = speakers["self_ic_calc"] >= speakers["self_ic_calc"].median()
    score_group = score_group.map(group_map)
    speakers["Score"] = score_group

def add_combined_ic_cefr(speakers):
    cefr_group = num_to_cefr(speakers["combined_calc"])
    speakers["Combined IC CEFR"] = cefr_group

def add_combined_score(speakers):
    group_map = {True: "Top Half", False: "Bottom Half"}
    score = speakers[["non_verbal_calc", "construct_calc", "self_ic_calc"]].mean(axis=1)
    speakers["combined_calc"] = score
    score_group = speakers["combined_calc"] >= speakers["combined_calc"].median()
    score_group = score_group.map(group_map)
    speakers["Score2"] = score_group

def get_score_corrs(speakers):
    cols = speakers.columns
    score_cols = ["non_verbal_calc", "self_ic_calc", "construct_calc"]
    scores = []
    for score_col in score_cols:
        if score_col in cols:
            scores.append(speakers[score_col])
    non_nans = np.full(len(speakers), True)
    series = []
    for score in scores:
        non_nans = non_nans & (~np.isnan(score))
    for score in scores:
        series.append(score[non_nans])
    corrs = np.corrcoef(np.array(series))
    return corrs, score_cols

def get_score_mismatch(speakers):
    n_start = len(speakers)
    mismatched_speakers = speakers[~(((speakers["Score"] == "Top Half") & (speakers["Score2"] == "Top Half")) | ((speakers["Score"] == "Bottom Half") & (speakers["Score2"] == "Bottom Half")))]
    n_end = len(mismatched_speakers)
    return n_end / n_start, mismatched_speakers

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
    columns_copy = columns.copy()
    columns_not_found = []
    for column in columns_copy:
        if column not in df.columns:
            columns_not_found.append(column)
    if len(columns_not_found) > 0:
        print(f"Columns not found: {len(columns_not_found)}")
    for column in columns_not_found:
        columns_copy.remove(column)
        print(f"\t{column}")
    items = len(columns_copy)
    cols = 5
    rows = math.ceil(items/cols)
    plt.figure(figsize=(15,3*rows))
    plt.tight_layout()
    for i, col in enumerate(columns_copy):
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

def find_fair_ratings(df, label, new_label):
    df = df.drop(index = df[df["rater"] == "R_19_old"].index)
    df[df == "R_19_new"] = "R_19"
    df = df.reset_index()
    
    mean_score = dict()
    std_score = dict()
    
    for rater in df.rater.value_counts().index.to_list():
        m = df[df["rater"] == rater][label].mean()
        std = df[df["rater"] == rater][label].std()
        mean_score[rater] = m
        std_score[rater] = std
    mean_m = np.array(list(mean_score.values())).mean()
    mean_std = np.array(list(std_score.values())).mean()

    df_normed = df.copy()

    for rater in df_normed.rater.value_counts().index.to_list():
        m = mean_score[rater]
        std = std_score[rater]
        df_normed.loc[df_normed["rater"] == rater, label] = ((df_normed[df_normed["rater"] == rater][label] - m + mean_m)/std) * mean_std

    mean_normed_m = df_normed[label].mean()

    speaker_scores = dict()
    for speaker in df_normed.speaker.value_counts().index.to_list():
        m = df_normed[df_normed["speaker"] == speaker][label].mean() - mean_normed_m + 0.5
        score = 5*m
        speaker_scores[int(speaker)] = score

    scores = pd.DataFrame(list(speaker_scores.values()), index=list(speaker_scores.keys()))
    scores.columns = [new_label]
    return scores

def combine_ratings(speakers, non_verbal_calc, construct_calc):
    return speakers.merge(non_verbal_calc, how="left", left_index=True, right_index=True).merge(construct_calc, how="left", left_index=True, right_index=True)
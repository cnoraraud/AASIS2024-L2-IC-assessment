import numpy as np
import pandas as pd
import io_tools as iot
import naming_tools as nt
import npz_reader as npzr
import npy_reader as npyr
import dataframe_reader as dfr

def clean_no_permissions(df):
    new_df = df
    perm1 = set(np.unique(df[df["Aasis study"] != 1]["SpeakerID"].values).tolist())
    perm2 = set(np.unique(df[df["Speech Permissions"] != 1]["SpeakerID"].values).tolist())
    perm3 = set(np.unique(df[df["Video Permissions"] != 1]["SpeakerID"].values).tolist())
    all_ids = perm1 or perm2 or perm3
    other_ids = set()
    for perm_id in all_ids:
        if perm_id % 2 == 0:
            other_ids.add(perm_id - 1)
        else:
            other_ids.add(perm_id + 1)
    all_ids.update(other_ids)
    for drop_id in all_ids:
        new_df = new_df[new_df["SpeakerID"] != drop_id]
    new_df = new_df.drop(columns=["Aasis study", "Later Contact", "Speech Permissions", "Video Permissions", "Storage Permissions"])
    return new_df

def process_names(names):
    dates = []
    speaker1s = []
    speaker2s = []
    tasks = []
    annotators = []
    for row in names:
        row_split = row.split("_")
        dates.append(pd.Timestamp(row_split[0], tz="Europe/Helsinki"))
        speakers = row_split[1].split("speaker")[-2:]
        speaker1s.append(int(speakers[0]))
        speaker2s.append(int(speakers[1]))
        tasks.append(row_split[2].replace("task",""))
        annotators.append(row_split[-1].split(".")[0])
    return pd.DataFrame({"datetime": dates, "speaker_1": speaker1s, "speaker_2": speaker2s, "task": tasks, "annotator": annotators})
def get_sample_dataframe():
    names = pd.DataFrame()
    npzs = []
    npys = []
    npy_list = npyr.npy_list()
    for npz_name in npzr.npz_list():
        npzs.append(npz_name)
        found = False
        for npy_name in npy_list:
            if nt.file_swap(npz_name, all=False) == nt.file_swap(npy_name, all=False):
                npys.append(npy_name)
                found = True
                break
        if not found:
            npys.append("-")
    names["npz"] = npzs
    names["npy"] = npys
    df = process_names(names["npz"])
    return names.join(df)

def get_clean_speaker_dataframe():
    return clean_no_permissions(get_speaker_dataframe())

def get_speaker_dataframe():
    spc_path = iot.special_data_path()
    spc_csvs = iot.list_dir(spc_path,"csv")
    unis = pd.read_csv(spc_path/'universities.csv',sep=";")
    ratings = pd.read_csv(spc_path/'ratings.csv',sep=";")
    speaker_info = pd.read_csv(spc_path/'consentquiz.csv',sep=";",encoding='latin-1')
    
    tasks = ratings["task"]
    just_ratings = ratings.drop(["task"], axis=1)
    means = just_ratings.mean(axis=1)
    ratings_by_speaker = dict()
    for task, mean in zip(list(tasks), means):
        task_values = task.split(" ")
        num_values = []
        other_values = []
        for value in task_values:
            if value.isnumeric():
                num_values.append(int(value))
            else:
                other_values.append(value)
        speaker = num_values[0]
        feature = ":".join(other_values)
        if speaker not in ratings_by_speaker:
            ratings_by_speaker[speaker] = dict()
        ratings_by_speaker[speaker][feature] = mean
    new_ratings = pd.DataFrame(ratings_by_speaker).T.reset_index()
    cols = list(new_ratings.columns)
    cols[0] = "SpeakerID"
    new_ratings.columns = cols

    return unis.merge(speaker_info, on="SpeakerID", how="outer").merge(new_ratings, on="SpeakerID", how="outer")

def get_cefr_ratings_dataframe():
    cefr = pd.read_csv(iot.ratings_csvs_path() / 'cefr_ratings.csv')
    cefr[cefr == -9999] = np.nan
    cefr[cefr == 9999] = np.nan
    cefr["construct_score"] = (cefr[["holistic", "fluency", "turn_taking"]].mean(axis=1) - 1)/5
    return cefr

def get_nonverbal_ratings_dataframe():
    non_verb = pd.read_csv(iot.ratings_csvs_path() / 'non_verbal_ratings.csv')
    non_verb[non_verb == 9999] = np.nan
    non_verb[non_verb == 100] = np.nan
    non_verb["if_contributed"] = (non_verb["if_contributed"] + 1) / 2
    non_verb["IC_score"] = non_verb[["face", "eye_contact", "head", "vocalizing", "hands", "if_contributed"]].mean(axis=1)
    return non_verb

def get_overall_ratings_dataframe():
    return pd.read_csv(iot.ratings_csvs_path() / 'overall_ratings.csv')

def get_speakers_full_dataframe():
    speakers = get_clean_speaker_dataframe()
    samples = get_sample_dataframe()
    non_verb = get_nonverbal_ratings_dataframe()
    cefr = get_cefr_ratings_dataframe()

    dfr.add_partner_id(speakers, samples)
    dfr.add_language_counts(speakers)
    dfr.add_language_difference(speakers)
    dfr.add_partner_relation(speakers)
    dfr.add_ic_score(speakers)

    non_verbal_calc = dfr.find_fair_ratings(non_verb, "IC_score", "non_verbal_calc")
    construct_calc = dfr.find_fair_ratings(cefr, "construct_score", "construct_calc")

    speakers = dfr.combine_ratings(speakers, non_verbal_calc, construct_calc)
    dfr.add_combined_score(speakers)
    dfr.add_combined_ic_cefr(speakers)

    return speakers
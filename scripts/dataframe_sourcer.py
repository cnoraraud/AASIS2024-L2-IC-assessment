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
    speaker_info = pd.read_csv(spc_path/'consentquiz.csv',sep=";",encoding='latin-1')

    return unis.merge(speaker_info, on="SpeakerID", how="outer")

def get_cefr_ratings_dataframe():
    cefr = pd.read_csv(iot.ratings_csvs_path() / 'cefr_ratings.csv')
    cefr[cefr == -9999] = np.nan
    cefr[cefr == 9999] = np.nan
    cefr["construct_score"] = cefr[["holistic", "fluency", "turn_taking"]].mean(axis=1)
    return cefr

def get_nonverbal_ratings_dataframe():
    non_verb = pd.read_csv(iot.ratings_csvs_path() / 'non_verbal_ratings.csv')
    non_verb[non_verb == 9999] = np.nan
    non_verb[non_verb == 100] = np.nan
    non_verb[non_verb == -9999] = np.nan
    non_verb[non_verb == -100] = np.nan
    non_verb["IC_score"] = non_verb[["face", "head", "vocalizing", "hands"]].mean(axis=1)
    return non_verb

def get_overall_ratings_dataframe():
    return pd.read_csv(iot.ratings_csvs_path() / 'overall_ratings.csv')

def select_for_task(df, tasks=[]):
    if len(tasks) == 0:
        return df
    cond = df["task"] == tasks[0]
    for i in range(1,len(tasks)):
        new_cond = df["task"] == tasks[i]
        cond = cond | new_cond
    return df[cond]

def ratings_to_speakers(df, tasks):
    return select_for_task(df, tasks).drop(columns=["task"]).groupby(by="speaker").mean().reset_index().rename(columns={"speaker":"SpeakerID"})

def get_rater_dict(ratings, target_columns, anchors_only=True):
    anchor_speakers = [20, 21, 35, 36, 87, 88]

    speakers = ratings["speaker"].unique()
    rater_dict = dict() 
    for speaker in speakers:
        if anchors_only and speaker not in anchor_speakers:
            continue
        speaker_ratings = ratings[ratings["speaker"] == speaker]
        mean_ratings = speaker_ratings[target_columns].mean()
        dif = speaker_ratings[target_columns] - mean_ratings
        raters = speaker_ratings.loc[dif.index][["rater","speaker"]]
        dif_w_raters = dif.merge(raters, left_index=True, right_index=True)
        for i, row in dif_w_raters.iterrows():
            rater = row["rater"]
            rater_difs = row[["speaker"] + target_columns]
            if rater not in rater_dict:
                rater_dict[rater] = []
            rater_dict[rater].append(rater_difs)
    return rater_dict

def get_adjusted_ratings(ratings, target_columns, ignore_columns, anchors_only=True):
    rater_dict = get_rater_dict(ratings, target_columns, anchors_only=anchors_only)
    
    df_adjusted = ratings.copy().drop(columns = ignore_columns)
    for rater in rater_dict.keys():
        adjustement = pd.DataFrame(rater_dict[rater]).groupby(by="speaker").mean().mean()
        indexes = df_adjusted[df_adjusted["rater"] == rater].index
        df_adjusted.loc[indexes, target_columns] -= adjustement

    df_adjusted_ratings = df_adjusted.drop(columns = ["rater"]).groupby(by=["task","speaker"]).mean().reset_index()
    return df_adjusted_ratings

non_verbal_columns = ["face", "eye_contact", "head", "vocalizing", "if_contributed"]
def get_speaker_nonverbals(tasks, anchors_only=True):
    nonverbal_adjusted_ratings = get_adjusted_ratings(get_nonverbal_ratings_dataframe(), non_verbal_columns, ["question_ind", "other_comment","feature_comment","impact_comment","one_row"], anchors_only=anchors_only)
    return ratings_to_speakers(nonverbal_adjusted_ratings, tasks), nonverbal_adjusted_ratings

cefr_columns = ["holistic", "range", "accuracy", "fluency", "pronunciation", "turn_taking"]
def get_speaker_cefrs(tasks, anchors_only=True):
    cefr_adjusted_ratings = get_adjusted_ratings(get_cefr_ratings_dataframe(), cefr_columns, ["task_type", "question_ind","comment","one_row"], anchors_only=anchors_only)
    return ratings_to_speakers(cefr_adjusted_ratings, tasks), cefr_adjusted_ratings

def get_speakers_samples_full_dataframe(tasks, anchors_only=True):
    speakers = get_clean_speaker_dataframe()
    samples = get_sample_dataframe()
    samples["task"] = samples["task"].str.lower()
    samples = select_for_task(samples, tasks=nt.task_names_no_prefix(tasks))

    non_verbal_speaker, non_verbal_adjusted = get_speaker_nonverbals(nt.task_names_no_prefix(tasks), anchors_only=anchors_only)
    cefr_calc_speaker, cefr_adjusted = get_speaker_cefrs(nt.task_names_no_prefix(tasks), anchors_only=anchors_only)

    dfr.add_partner_id(speakers, samples)
    dfr.add_language_counts(speakers)
    dfr.add_language_difference(speakers)
    dfr.add_partner_relation(speakers)
    dfr.add_ic_score(speakers)

    speakers = dfr.combine_speakers_ratings(speakers, [non_verbal_speaker, cefr_calc_speaker])
    samples = dfr.combine_samples_ratings(samples, [non_verbal_adjusted, cefr_adjusted])

    dfr.add_CEFR_scores(speakers, ["holistic"])

    return speakers, samples
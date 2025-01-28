import numpy as np
from datetime import date
import pandas as pd
import io_tools as iot
import npz_reader as npzr

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
        speaker1s.append(speakers[0])
        speaker2s.append(speakers[1])
        tasks.append(row_split[2].replace("task",""))
        annotators.append(row_split[-1].split(".")[0])
    return pd.DataFrame({"datetime": dates, "speaker_1": speaker1s, "speaker_2": speaker2s, "task": tasks, "annotator": annotators})
def get_sample_dataframe():
    names = pd.DataFrame()
    names["name"] = npzr.npz_list()
    df = process_names(names["name"])
    return names.join(df)

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
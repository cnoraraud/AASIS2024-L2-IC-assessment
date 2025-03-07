import os
import sys
import yaml
import shutil
import pathlib as p
from datetime import datetime
import numpy as np
from pympi import Elan as elan
import elan_tools as elant
import name_tools as namet

def get_config_private():
    with open('../config/privateconfig.yml', 'r') as file:
        return yaml.safe_load(file)
def get_config_public():
    with open('../config/publicconfig.yml', 'r') as file:
        return yaml.safe_load(file)

def list_dir(path, extension="*"):
    names = []
    for file in path.glob(f'**/*.{extension}'):
        names.append(file.name)
    return names

def annotated_list():
    config = get_config_private()
    aasis_path = p.Path(config["paths"]["main_data"]["."])
    eafs_path = aasis_path / config["paths"]["main_data"]["eafs"]
    annotation_list = []
    for eaf in eafs_path.glob('**/*.eaf'):
        annotation = eaf.stem
        if annotation not in annotation_list:
            annotation_list.append(annotation)
    return annotation_list

def get_csv_names_joystick(eaf_name):
    annotators = ["il","jk"]
    versions = ["001", "002"]
    base_name = eaf_name
    if "take" in eaf_name:
        base_name = "_".join(eaf_name.split("_")[:-1])
    speakers = base_name.split("_")[1].split("speaker")[1:]
    csv_names = []
    for speaker in speakers:
        for annotator in annotators:
            for version in versions:
                csv_name = f"{base_name}_speaker{speaker}_{annotator}_{version}.csv"
                csv_names.append(csv_name)
    return csv_names

def private_data_path(key):
    config = get_config_private()
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    data_path = personal_path / config["paths"]["personal_data"][key]
    return data_path

def aasis_data_path(key):
    config = get_config_private()
    aasis_path = p.Path(config["paths"]["main_data"]["."])
    data_path = aasis_path / config["paths"]["main_data"][key]
    return data_path

def aasis_wavs_path():
    return aasis_data_path("wavs")

def aasis_eafs_path():
    return aasis_data_path("eafs")

def aasis_csvs_path():
    return aasis_data_path("csvs")

def aasis_mp4s_path():
    return aasis_data_path("mp4s")

def wavs_path():
    return private_data_path("wavs")

def eafs_path():
    return private_data_path("eafs")

def csvs_path():
    return private_data_path("csvs")

def npzs_path():
    return private_data_path("npzs")

def npys_path():
    return private_data_path("npys")

def special_data_path():
    return private_data_path("special_data")

def figs_path():
    return private_data_path("figs")

# Pref use source_data_from_sessions
def source_annotated_data():
    config = get_config_private()
    aasis_path = p.Path(config["paths"]["main_data"]["."])
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    eafs_src_path = aasis_eafs_path()
    wavs_src_path = aasis_wavs_path()
    csvs_src_path = aasis_csvs_path()
    eafs_dst_path = eafs_path()
    wavs_dst_path = wavs_path()
    csvs_dst_path = csvs_path()
    
    with open(personal_path / "copy_manifest.txt", "a") as manifest:
        manifest.write(f"\nSTARTING copy {datetime.now()}\n")
        version = config["version"]
        date = config["date"]
        manifest.write(f"\tCONFIG: version {version} from {date}\n")
        manifest.write(f"\tFROM: {aasis_path} TO: {personal_path}\n")
        good = 0
        new = 0
        bad = 0
        get_choice = -1
        for annotation in annotated_list():
            try:
                # find eaf
                eaf_annotation = "_".join(annotation.split("_")[:-1])
                eaf_src_choices = sorted(eafs_src_path.glob(f'**/{eaf_annotation}*.eaf'))
                if len(eaf_src_choices) == 0:
                    manifest.write(f"\t{annotation}: no eaf choices\n")

                eaf_path = eaf_src_choices[get_choice]
                # find wavs
                wav_names, mp4_names = elant.get_wav_names(elan.Eaf(eaf_path))
                wavs = []
                for wav_name in wav_names:
                    if not p.Path(wavs_dst_path / wav_name).exists():
                        wav_src_choices = sorted(wavs_src_path.glob(f'**/{wav_name}')) # lazy
                        if len(wav_src_choices) > 0:
                            wav = wav_src_choices[get_choice]
                            wavs.append(wav)
                        else:
                            manifest.write(f"\t{annotation}: wav {wav_name} not found\n")
                # find csvs possibilities
                csv_names = get_csv_names_joystick(eaf_annotation)
                csvs = []
                for csv_name in csv_names:
                    if not p.Path(csvs_dst_path / csv_name).exists():
                        csv_src_choices = sorted(csvs_src_path.glob(f'**/{csv_name}'))
                        if len(csv_src_choices) > 0:
                            csv = csv_src_choices[get_choice]
                            csvs.append(csv)
                        else:
                            pass
                            #manifest.write(f"\t{annotation}: csv {csv_name} not found\n")

                # copy
                updated = 0
                if not p.Path(eafs_dst_path / eaf_path.name).exists():
                    shutil.copy(eaf_path, eafs_dst_path / eaf_path.name)
                    manifest.write(f"\t{annotation}: eaf succeeded\n")
                    updated += 1
                for wav in wavs:
                    shutil.copy(wav, wavs_dst_path / wav.name)
                    manifest.write(f"\t{annotation}: wav succeeded\n")
                    updated += 1 
                for csv in csvs:
                    shutil.copy(csv, csvs_dst_path / csv.name)
                    manifest.write(f"\t{annotation}: csv succeeded\n")
                    updated += 1 
                
                if updated == 0:
                    manifest.write(f"\t{annotation}: already existed\n")
                else:
                    new += updated
                good += 1
            except Exception as e:
                manifest.write(f"\t{annotation}: failed\n\t\terror: {e}\n")
                bad += 1
        manifest.write(f"\tSTATS: good {good} bad {bad} new {new}\n")
        manifest.write(f"FINISHING copy {datetime.now()}\n")

def any_in(search_string, sub_strings):
    for sub_string in sub_strings:
        if sub_string in search_string:
            return True
    return False

def find_session_files(file_list, task, speakers):
    match_list = []
    for file_path in file_list:
        file_name = p.Path(file_path).name
        if task not in file_name:
            continue
        if not any_in(file_name, speakers):
            continue
        match_list.append(file_path)
    return match_list

def get_newest_file_paths(file_paths):
    file_names = set()
    for file_path in file_paths:
        file_names.add(file_path.name)
    file_names = sorted(list(file_names))
    output_file_paths = []
    for file_name in file_names:
        path_list = []
        time_list = []
        for file_path in file_paths:
            if file_name == file_path.name:
                path_list.append(file_path)
                time_list.append(os.path.getmtime(f"{file_path}"))
        path_list = np.array(path_list)
        time_list = np.array(time_list)
        path_list_sorted = path_list[np.argsort(time_list)].tolist()
        if len(path_list_sorted) > 0:
            output_file_paths.append(path_list_sorted[-1])
    return output_file_paths

def get_aasis_sessions():
    sessions = dict()
    eaf_list = sorted(aasis_eafs_path().glob(f'**/*.eaf'))
    eafs = get_newest_file_paths(eaf_list)
    wav_list = sorted(aasis_wavs_path().glob(f'**/*.wav'))
    csv_list = sorted(aasis_csvs_path().glob(f'**/*.csv'))
    mp4_list = sorted(aasis_mp4s_path().glob(f'**/*.mp4'))
    for eaf in eafs:
        eaf_name = eaf.name
        task = namet.find_task(eaf_name)
        speakers = namet.find_speakers(eaf_name)
        wavs = []
        csvs = []
        mp4s = []
        for wav_file_path in find_session_files(wav_list, task, speakers):
            wavs.append(wav_file_path)
        for csv_file_path in find_session_files(csv_list, task, speakers):
            csvs.append(csv_file_path)
        for mp4_file_path in find_session_files(mp4_list, task, speakers):
            mp4s.append(mp4_file_path)
        wavs = get_newest_file_paths(wavs)
        csvs = get_newest_file_paths(csvs)
        mp4s = get_newest_file_paths(mp4s)
        sessions[eaf_name] = {"name": eaf_name, "eaf":eaf, "wavs": wavs, "csvs": csvs, "mp4s":mp4s}
    return sessions

def print_aasis_sessions():
    print("Finding Sessions...")
    sessions = get_aasis_sessions()
    for session_key in sessions:
        session = sessions[session_key]
        print(f"{session["name"]} ({session["eaf"]})")
        print("\twavs:")
        for wav in session["wavs"]:
            print(f"\t\t{wav.name} ({wav})")
        print("\tcsvs:")
        for csv in session["csvs"]:
            print(f"\t\t{csv.name} ({csv})")
        print("\tmp4s:")
        for mp4 in session["mp4s"]:
            print(f"\t\t{mp4.name} ({mp4})")
        print()

def source_data_from_sessions(sessions):
    #TODO
    pass

if __name__ == '__main__':
    globals()[sys.argv[1]]()
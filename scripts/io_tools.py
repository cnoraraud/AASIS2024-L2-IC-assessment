import os
import sys
import yaml
import shutil
import traceback
import pathlib as p
from datetime import datetime
import numpy as np
from pympi import Elan as elan
import elan_tools as elant
import naming_tools as nt
import time
import data_logger as dl

def find_config_directory(n = 3):
    path_string = './config/'
    for i in range(n):
        dir = p.Path(path_string)
        if dir.exists() and dir.is_dir():
            return dir
        path_string = "." + path_string
    raise IOError("Config folder not found upstream from script!") 

def get_config_private():
    with open((find_config_directory() / 'privateconfig.yml'), 'r') as file:
        return yaml.safe_load(file)

def get_config_public():
    with open((find_config_directory() / 'publicconfig.yml'), 'r') as file:
        return yaml.safe_load(file)

def get_config_version():
    config = get_config_private()
    config_version = config["version"]
    return config_version

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

def aasis_pyfeat_csvs_path():
    return aasis_data_path("pyfeat_csvs")

def aasis_joystick_csvs_path():
    return aasis_data_path("joystick_csvs")

def aasis_ratings_csvs_path():
    return aasis_data_path("ratings_csvs")

def wavs_path(sr = None):
    if isinstance(sr, type(None)):
        return private_data_path("wavs")
    return private_data_path("wavs") / f"sr{sr}"

def eafs_path():
    return private_data_path("eafs")

def csvs_path():
    return private_data_path("csvs")

def mp4s_path():
    return private_data_path("mp4s")

def pyfeat_csvs_path():
    return private_data_path("pyfeat_csvs")

def joystick_csvs_path():
    return private_data_path("joystick_csvs")

def ratings_csvs_path():
    return private_data_path("ratings_csvs")

def npzs_path():
    return private_data_path("npzs")

def npys_path():
    return private_data_path("npys")

def special_data_path():
    return private_data_path("special_data")

def figs_path():
    return private_data_path("figs")

def output_csvs_path():
    return private_data_path("output_csvs")

def manifests_path():
    return private_data_path("manifests")

def list_dir(path, extension="*"):
    names = []
    for file in path.glob(f'**/*.{extension}'):
        names.append(file.name)
    return names

def read_metadata_from_path(path):
    name = "unknown_name"
    cmtime = "unknown_modification_time"
    size = "unknown_size"
    type = "unknown_type"
    exists = os.path.exists(path)
    if exists:
        name = nt.get_name(path)
        mtime = os.path.getmtime(f"{path}")
        cmtime = str(time.ctime(mtime))
        size = os.path.getsize(path)
        if os.path.isdir(path):
            type = "dir"
        elif os.path.isfile(path):
            type = "file"

    return {"name": name, "mtime": cmtime, "size": size, "type": type}

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

def get_csv_names_facial_features(name):
    pyfeat_csv_list = pyfeat_csvs_path().glob(f'**/*.csv')
    session = get_related_session_for_file(name, use_defaults=False, pyfeat_csv_list=pyfeat_csv_list)
    return session["pyfeat_csvs"]

def get_wav_paths(name):
    wavs_list = wavs_path().glob(f'**/*.wav')
    session = get_related_session_for_file(name, use_defaults=False, wav_list=wavs_list)
    return session["wavs"]

def get_csv_paths_joystick_discrete(eaf_name):
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

def get_csv_paths_joystick(name):
    joystick_csv_list = joystick_csvs_path().glob(f'**/*.csv')
    session = get_related_session_for_file(name, use_defaults=False, joystick_csv_list=joystick_csv_list)
    return session["joystick_csvs"]

#@deprecated("Use fuzzy sourcing instead")
def source_annotated_data_discrete():
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
        manifest.write(f"\nSTARTING discrete copy {datetime.now()}\n")
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
                csv_names = get_csv_paths_joystick_discrete(eaf_annotation)
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
                eaf = p.Path(eafs_dst_path / eaf_path.name)
                if not eaf.exists():
                    shutil.copy(eaf_path, eaf)
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

def check_configured(key_name, key_value):
    dl.log(key_value)
    if "<" in key_value or ">" in key_value:
        dl.log(f"Key \'{key_name}\' not configured in! (\'{key_value}\')")
        raise NameError("")

def create_wavs_sr_folder(sr):
    if not wavs_path(sr).exists():
        wavs_path(sr).mkdir(parents=True, exist_ok=False)

def create_data_folders():
    new_folders = 0
    successful = False
    try:
        dl.log("Attempting automatic data folder creation.")
        config = get_config_private()
        personal_data = config["paths"]["personal_data"]
        rootdir = personal_data["."]
        check_configured(".", rootdir)
        rootdir_path = p.Path(rootdir)
        if create_missing_folder(rootdir_path): new_folders += 1
        subdirs = list(personal_data.keys())
        subdirs.remove(".")
        for subdir_key in subdirs:
            subdir = personal_data[subdir_key]
            check_configured(f"subdir:{subdir_key}", subdir)
            subdir_path = rootdir_path / p.Path(subdir)
            if create_missing_folder(subdir_path): new_folders += 1
        dl.log("Automatic data folder creation succeeded.")
        successful = True
    except NameError as ve:
        dl.log("Automatic data folder creation failed.")
        dl.log("\tEncountered NameError in creating folders. Has \'privateconfig.yml\' been created and configured?")
    except Exception as e:
        dl.log("Automatic data folder creation failed.")
        dl.log(traceback.format_exc())
    finally:
        dl.log(f"Total new folders: {new_folders}")
    return successful

def create_missing_folder(tgt):
    if p.Path(tgt).exists():
        return False
    os.mkdir(tgt)
    return True

def create_missing_folder_recursive(tgt):
    if p.Path(tgt).exists():
        return False
    os.makedirs(tgt)
    return True

def copy_missing(src, tgt, overwrite=False):
    if p.Path(tgt).exists() and not overwrite:
        return False
    shutil.copy(src, tgt)
    return True

def remove_plural(word):
    if len(word) == 0: return word
    if word[-1] == "s": return word[:-1]
    return word

def source_annotated_data_fuzzy(overwrite=False):
    manifest = dl.ManifestSession("copy")
    manifest.start()

    sessions = get_aasis_sessions()
    for session_key in sessions:
        try:
            session = sessions[session_key]
            manifest.session_start(session["name"])

            tags = ["eafs", "wavs", "csvs", "pyfeat_csvs", "joystick_csvs"]
            for tag in tags:
                tag_name = remove_plural(tag)
                file_paths = session[tag]
                file_dst_path = private_data_path(tag)
                for file_path in file_paths:
                    if copy_missing(file_path, file_dst_path / file_path.name, overwrite=overwrite):
                        manifest.new_file(tag_name, file_path.name)
            
            manifest.session_end()
        except Exception as e:
            dl.log(traceback.format_exc())
            manifest.error(e)
    manifest.end()

def source_aasis_ratings(overwrite=True):
    ratings_csv_list = sorted(aasis_ratings_csvs_path().glob(f'**/*.csv'))
    for file_path in ratings_csv_list:
        new_path = ratings_csvs_path() / file_path.name
        if copy_missing(file_path, new_path, overwrite=overwrite):
            dl.write_to_manifest_new_file(dl.COPY_TYPE, file_path, new_path)

def any_in(search_string, sub_strings):
    for sub_string in sub_strings:
        if sub_string in search_string:
            return True
    return False

def find_session_files(file_list=None, task=None, speakers=None):
    if isinstance(file_list, type(None)):
        file_list = []
    
    match_list = []
    for file_path in file_list:
        file_name = p.Path(file_path).name
        if task not in file_name: continue
        if not any_in(file_name, speakers): continue

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

# TODO (low prio): Generics somehow...
def get_related_session_for_file(file, use_defaults=True, eaf_list=None, wav_list=None, csv_list=None, mp4_list=None, pyfeat_csv_list=None, joystick_csv_list=None):
    # Defaults
    file_path = p.Path(file)
    file_name = file_path.name
    if use_defaults:
        if isinstance(eaf_list, type(None)):
            eaf_list = sorted(eafs_path().glob(f'**/*.eaf'))
        if isinstance(wav_list, type(None)):
            wav_list = sorted(wavs_path().glob(f'**/*.wav'))
        if isinstance(csv_list, type(None)):
            csv_list = sorted(csvs_path().glob(f'**/*.csv'))
        if isinstance(mp4_list, type(None)):
            mp4_list = sorted(mp4s_path().glob(f'**/*.mp4'))
        if isinstance(pyfeat_csv_list, type(None)):
            pyfeat_csv_list = sorted(pyfeat_csvs_path().glob(f'**/*.csv'))
        if isinstance(joystick_csv_list, type(None)):
            joystick_csv_list = sorted(joystick_csvs_path().glob(f'**/*.csv'))

    # Session
    task = nt.find_task(file_name)
    speakers = nt.find_speakers(file_name)
    
    eafs = []
    wavs = []
    csvs = []
    mp4s = []
    pyfeat_csvs = []
    joystick_csvs = []

    for eaf_file_path in find_session_files(eaf_list, task, speakers):
        eafs.append(eaf_file_path)
    for wav_file_path in find_session_files(wav_list, task, speakers):
        wavs.append(wav_file_path)
    for csv_file_path in find_session_files(csv_list, task, speakers):
        csvs.append(csv_file_path)
    for mp4_file_path in find_session_files(mp4_list, task, speakers):
        mp4s.append(mp4_file_path)
    for pyfeat_csv_path in find_session_files(pyfeat_csv_list, task, speakers):
        pyfeat_csvs.append(pyfeat_csv_path)
    for joystick_csv_path in find_session_files(joystick_csv_list, task, speakers):
        joystick_csvs.append(joystick_csv_path)

    eafs = get_newest_file_paths(eafs)
    wavs = get_newest_file_paths(wavs)
    csvs = get_newest_file_paths(csvs)
    mp4s = get_newest_file_paths(mp4s)
    pyfeat_csvs = get_newest_file_paths(pyfeat_csvs)
    joystick_csvs = get_newest_file_paths(joystick_csvs)
    return {"name": file_name,
            "eafs": eafs,
            "wavs": wavs,
            "csvs": csvs,
            "mp4s": mp4s,
            "pyfeat_csvs": pyfeat_csvs,
            "joystick_csvs": joystick_csvs}

def get_sessions_from_lists(eaf_list, wav_list, csv_list, mp4_list, pyfeat_csv_list, joystick_csv_list):
    sessions = dict()
    eafs = get_newest_file_paths(eaf_list)
    for eaf in eafs:
        session = get_related_session_for_file(
            eaf,
            eaf_list=eaf_list,
            wav_list=wav_list,
            csv_list=csv_list,
            mp4_list=mp4_list,
            pyfeat_csv_list=pyfeat_csv_list,
            joystick_csv_list=joystick_csv_list)
        sessions[session["name"]] = session
    return sessions

def get_sessions():
    eaf_list = sorted(eafs_path().glob(f'**/*.eaf'))
    wav_list = sorted(wavs_path().glob(f'**/*.wav'))
    csv_list = sorted(csvs_path().glob(f'**/*.csv'))
    mp4_list = sorted(mp4s_path().glob(f'**/*.mp4'))
    pyfeat_csv_list = sorted(pyfeat_csvs_path().glob(f'**/*.csv'))
    joystick_csv_list = sorted(joystick_csvs_path().glob(f'**/*.csv'))
    return get_sessions_from_lists(eaf_list, wav_list, csv_list, mp4_list, pyfeat_csv_list, joystick_csv_list)

def get_aasis_sessions():
    eaf_list = sorted(aasis_eafs_path().glob(f'**/*.eaf'))
    wav_list = sorted(aasis_wavs_path().glob(f'**/*.wav'))
    csv_list = sorted(aasis_csvs_path().glob(f'**/*.csv'))
    mp4_list = sorted(aasis_mp4s_path().glob(f'**/*.mp4'))
    pyfeat_csv_list = sorted(aasis_pyfeat_csvs_path().glob(f'**/*.csv'))
    joystick_csv_list = sorted(aasis_joystick_csvs_path().glob(f'**/*.csv'))
    return get_sessions_from_lists(eaf_list, wav_list, csv_list, mp4_list, pyfeat_csv_list, joystick_csv_list)

def print_sessions():
    dl.log("Finding Local Sessions...")
    sessions = get_sessions()
    print_from_sessions(sessions)

def print_aasis_sessions():
    dl.log("Finding Aasis Sessions...")
    sessions = get_aasis_sessions()
    print_from_sessions(sessions)

def print_from_sessions(sessions):
    for session_key in sessions:
        session = sessions[session_key]
        name = session["name"]
        dl.log(f"{name}")
        for file_key in session:
            file_paths = session[file_key]
            if not isinstance(file_paths, list):
                continue
            dl.log(f"\t{file_key}:")
            for file_path in file_paths:
                dl.log(f"\t\t{file_path.name} ({file_path.parents[0]})")
        dl.log()

if __name__ == '__main__':
    globals()[sys.argv[1]]()
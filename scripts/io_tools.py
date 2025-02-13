import sys
import yaml
import pathlib as p
from datetime import datetime
import shutil
import elan_tools
from pympi import Elan as elan

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

def source_annotated_data():
    config = get_config_private()
    aasis_path = p.Path(config["paths"]["main_data"]["."])
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    eafs_src_path = aasis_path / config["paths"]["main_data"]["eafs"]
    wavs_src_path = aasis_path / config["paths"]["main_data"]["wavs"]
    csvs_src_path = aasis_path / config["paths"]["main_data"]["csvs"]
    eafs_dst_path = eafs_path()
    wavs_dst_path = wavs_path()
    csvs_dst_path = csvs_path()
    
    with open(personal_path / "copy_manifest.txt", "a") as manifest:
        manifest.write(f"\nSTARTING copy {datetime.now()}\n")
        manifest.write(f"\tCONFIG: version {config["version"]} from {config["date"]}\n")
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
                eaf_path = eaf_src_choices[get_choice]
                # find wavs
                wav_names = elan_tools.get_wav_names(elan.Eaf(eaf_path))
                wavs = []
                for wav_name in wav_names:
                    if not p.Path(wavs_dst_path / wav_name).exists():
                        wav_src_choices = sorted(wavs_src_path.glob(f'**/{wav_name}')) # lazy
                        wav = wav_src_choices[get_choice]
                        wavs.append(wav)
                # find csvs possibilities
                csv_names = get_csv_names_joystick(eaf_annotation)
                csvs = []
                for csv_name in csv_names:
                    if not p.Path(csvs_dst_path / csv_name).exists():
                        csv_src_choices = sorted(csvs_src_path.glob(f'**/{csv_name}'))
                        if len(csv_src_choices) > 0:
                            csv = csv_src_choices[get_choice]
                            csvs.append(csv)

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

if __name__ == '__main__':
    globals()[sys.argv[1]]()
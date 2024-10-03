import yaml
import pathlib as p
from os import listdir, walk
from datetime import datetime
import shutil

def get_config_private():
    with open('../config/privateconfig.yml', 'r') as file:
        return yaml.safe_load(file)
def get_config_public():
    with open('../config/publicconfig.yml', 'r') as file:
        return yaml.safe_load(file)

def annotated_list():
    config = get_config_private()
    aasis_path = p.Path(config["paths"]["main_data"]["."])
    eafs_path = aasis_path / config["paths"]["main_data"]["eafs"]
    annotation_list = []
    for eaf in eafs_path.iterdir():
        annotation = eaf.stem
        if annotation not in annotation_list:
            annotation_list.append(annotation)
    return annotation_list

def copy_annotated_data_from_aasis():
    config = get_config_private()
    aasis_path = p.Path(config["paths"]["main_data"]["."])
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    eafs_src_path = aasis_path / config["paths"]["main_data"]["eafs"]
    pfsxs_src_path = aasis_path / config["paths"]["main_data"]["pfsxs"]
    wavs_src_path = aasis_path / config["paths"]["main_data"]["wavs"]
    eafs_dst_path = personal_path / config["paths"]["personal_data"]["eafs"]
    pfsxs_dst_path = personal_path / config["paths"]["personal_data"]["pfsxs"]
    wavs_dst_path = personal_path / config["paths"]["personal_data"]["wavs"]
    
    with open(personal_path / "copy_manifest.txt", "a") as manifest:
        manifest.write(f"\nSTARTING copy {datetime.now()}\n")
        manifest.write(f"\tCONFIG: version {config["version"]} from {config["date"]}\n")
        manifest.write(f"\tFROM: {aasis_path} TO: {personal_path}\n")
        good = 0
        bad = 0
        for annotation in annotated_list():
            try:
                annotation_path = p.Path(annotation)
                eaf = annotation_path.with_suffix(".eaf")
                wav = annotation_path.with_suffix(".wav")
                pfsx = annotation_path.with_suffix(".pfsx")
                shutil.copy(eafs_src_path / eaf, eafs_dst_path)
                #shutil.copy(pfsxs_src_path / pfsx, pfsxs_dst_path)
                shutil.copy(wavs_src_path / wav, wavs_dst_path)

                manifest.write(f"\t{annotation}: succeeded\n")
                good += 1
            except:
                manifest.write(f"\t{annotation}: failed\n")
                bad += 1

        manifest.write(f"\tSTATS: good {good} bad {bad}\n")
        manifest.write(f"FINISHING copy {datetime.now()}\n")


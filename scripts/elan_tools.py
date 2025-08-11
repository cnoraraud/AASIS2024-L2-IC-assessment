import sys
import re
import pathlib as p
import pandas as pd
from pympi import Elan as elan
import io_tools as iot
import data_reader as dr
import data_logger as dl


def get_wav_names(eaf):
    wav_names = []
    mp4_names = []
    for descriptor in eaf.media_descriptors:
        if "audio" in descriptor["MIME_TYPE"]:
            url = None
            if "RELATIVE_MEDIA_URL" in descriptor:
                url = descriptor["RELATIVE_MEDIA_URL"]
            elif "MEDIA_URL" in descriptor:
                url = descriptor["MEDIA_URL"]  # :(
            if url is not None:
                wav_names.append(p.Path(url).name)
        if "video" in descriptor["MIME_TYPE"]:
            url = None
            if "RELATIVE_MEDIA_URL" in descriptor:
                url = descriptor["RELATIVE_MEDIA_URL"]
            elif "MEDIA_URL" in descriptor:
                url = descriptor["MEDIA_URL"]  # :(
            if url is not None:
                mp4_names.append(p.Path(url).name)
    return wav_names, mp4_names


def eaf_media_scrape():
    rows = []
    for eaf_path in dr.list_eaf()[0]:
        eaf_name = eaf_path.name
        row = {"eaf": eaf_name}
        eaf = elan.Eaf(eaf_path)
        wav_names, mp4_names = get_wav_names(eaf)
        # ws = sorted(wav_names)
        # ms = sorted(mp4_names)
        for wav_name in wav_names:
            key = "wavs:" + re.search(r"(mic(\d)+)+", wav_name)[0]
            row[key] = wav_name
        for mp4_name in mp4_names:
            key = "mp4s:" + re.search(r"(cam(\d)+)+", mp4_name)[0]
            row[key] = mp4_name
        rows.append(row)
    df = pd.DataFrame(rows)
    path = iot.special_data_path() / "eaf-to-wav-and-mp4.csv"
    df[sorted(df.columns)].to_csv(path)
    dl.log(f"Table of annotationa and media correspondence created at {path}")


if __name__ == "__main__":
    globals()[sys.argv[1]]()

import sys
import re
import pathlib as p

import numpy as np
from scipy.io import wavfile
from pympi import Elan as elan

import io_tools
import elan_tools
import wav_tools

def num_to_n_chars(num:int, n:int):
    return str(num).rjust(n, "0")[-n:]

def ropt(r_):
    return r"(" + r_ + ")?"

def list_eaf(year:int=None, month:int=None, day:int=None, speaker:int=None, speaker2:int=None, task=None, set:int=None, extension=None):
    # default wildcard regex
    rs = r"_"
    ryear = r"\d{4}"
    rmonth = r"\d{2}"
    rday = r"\d{2}"
    rspeaker = r"speaker\d{3}"
    rspeaker2 = ropt(rspeaker)
    rtask = r"task\d{1}[AB]?"
    rset = r"set\d{2}"
    rtake = r"take\d{1}"
    rauthor = r"\w+"
    rextension = r"\..{2,4}"

    # input regex
    if year is not None:
        ryear = num_to_n_chars(year, 4)
    if month is not None:
        rmonth = num_to_n_chars(month, 2)
    if day is not None:
        rday = num_to_n_chars(day, 2)
    if speaker is not None:
        rspeaker = r"speaker" + num_to_n_chars(speaker, 3)
    if speaker2 is not None:
        rspeaker2 = r"speaker" + num_to_n_chars(speaker2, 3)
    if task is not None:
        if type(task) == int:
            rtask = r"task" + num_to_n_chars(task, 1)
        else:
            rtask = r"task" + task
    if set is not None:
        rset =  r"set" + num_to_n_chars(set, 2)
    if extension is not None:
        rextension = extension.replace(".", "")
        rextension = r"\." + rextension
    
    rextra = ropt(rs + rset) + ropt(rs + rtake) + ropt(rs + rauthor) + ropt(rextension)

    regex = re.compile(ryear + rmonth + rday + rs 
                       + rspeaker + rspeaker2 + rs
                       + rtask + rextra)
    return list_eaf_with_regex(regex)

def check_data_row(table, key):
    if key not in table:
        row = dict()
        row["wavs"] = []
        row["eafpath"] = None
        row["eaf"] = None
        table[key] = row

def list_eaf_with_regex(regex):
    # wav, eaf naming convention: <4: year><4: recording>_speaker<3: speaker>_task<1: task>_set<2: set>.<extension>
    config = io_tools.get_config_private()
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    eafs_path = personal_path / config["paths"]["personal_data"]["eafs"]
    eafs_path_list = []
    data_table = dict()
    for eaf in eafs_path.iterdir():
        if regex.match(eaf.name):
            key = eaf.stem
            check_data_row(data_table, key)
            data_table[key]["eafpath"] = eaf
            eafs_path_list.append(eaf)
    
    return sorted(eafs_path_list), data_table

def data_head(table, n = None):
    return data_section(table, n)

def data_section(table, start = 0, n = None):
    if n is None:
        n = len(table) - start
    return dict(list(table.items())[start : start + n]) # dictionaries are ordered as of python 3.7

def get_r_name(r, id, wav_name):
    name_stem = wav_name.split(".")[0]
    match = re.search(r, name_stem)
    group = match.group()
    uncleaned = tuple(group.split(id)[1:])
    return uncleaned

def get_speaker_name(wav_name):
    rspeaker = r"(speakers?([\d])+)+"
    return get_r_name(rspeaker, "speaker", wav_name)

def get_mic_name(wav_name):
    rmic = r"(mics?([0-9a-z])+)+"
    return get_r_name(rmic, "mic", wav_name)

def read_wav(wav_names, wavs_path_sr, wavs_path, sr):
    wavs = []
    for wav_name in wav_names:
        try:
            wav_path_sr = wavs_path_sr / wav_name
            wav_path = wavs_path / wav_name
            wav = None
            sr0 = None
            data0 = None
            speakers0 = get_speaker_name(wav_name)
            mics0 = get_mic_name(wav_name)
            if not wavs_path_sr.exists():
                wavs_path_sr.mkdir(parents=True, exist_ok=False)
            
            if wav_path_sr.exists():
                sr0, data0 = wavfile.read(wav_path_sr.as_posix())
                wav = (sr0, data0, speakers0, mics0)
            elif wav_path.exists():
                sr0, data0 = wav_tools.resample_wavs(wav_path = wav_path, wav_path_sr = wav_path_sr, sr = sr)
                wav = (sr0, data0, speakers0, mics0)
            
            if wav is not None:
                wavs.append(wav)

        except Exception as e:
            print(f"Could not read {wav_name} due to {e}")
    return wavs

def read_row(eafpath, sr:int=None, do_wav = False):
    config = io_tools.get_config_private()
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    wav_names = []
    mp4_names = []

    # populate eaf
    try:
        eaf = elan.Eaf(eafpath)
        wav_names, mp4_names = elan_tools.get_wav_names(eaf)
    except:
        print(f"Could not read {eafpath}")
    
    wavs = []
    if do_wav:
        wavs_path = personal_path / config["paths"]["personal_data"]["wavs"]
        wavs_path_sr = wavs_path if sr is None else wavs_path / f"sr{sr}/"
        wavs = read_wav(wav_names, wavs_path_sr = wavs_path_sr, wavs_path = wavs_path, sr = sr)
        
    return eaf, wavs

def read_rows(table, key_whitelist=None, sr:int=None, do_wavs = False):
    config = io_tools.get_config_private()
    personal_path = p.Path(config["paths"]["personal_data"]["."])
    wavs_path = personal_path / config["paths"]["personal_data"]["wavs"]
    wavs_path_sr = wavs_path if sr is None else wavs_path / f"sr{sr}/"
    
    all_keys = []
    all_rows = []
    all_wavs = []
    for key in table:
        if key_whitelist is None or key in key_whitelist:
            row = table[key]
            eaf, wavs = read_row(eafpath = row["eafpath"], sr = sr, do_wav = do_wavs)
            row["eaf"] = eaf
            all_keys.append(key)
            all_rows.append(row)
            all_wavs.append(wavs)
    return all_keys, all_rows, all_wavs

def get_rows(table = None, n = None, start:int = 0, sr:int = None, do_wavs = False):
    if table is None:
        eafs, table = list_eaf()
    return read_rows(table = data_section(table, start, n), sr = sr, do_wavs = do_wavs)

def test_eaf_reading():
    eafs, table = list_eaf()
    for eaf in eafs:
        print(eaf)
    print(f"eaf count {len(eafs)}")

class DataProvider:
    eafs = []
    table = dict()

    def __len__(self):
        return len(self.eafs)

    def __init__(self, sr = 16000):
        self.sr = sr
        self.eafs, self.table = list_eaf()
    
    def reset(self):
        self.pointer = 0

    def generator(self, do_wavs = False, n = 1):
        i = 0
        while i < len(self):
            yield get_rows(self.table, n = n, start = i, sr = self.sr, do_wavs = do_wavs)
            i += n
    
    def nth_key(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        return list(rows.keys())[n]
    
    def nth(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        return rows[self.nth_key(rows, n)]

    def report_rows(self, rows = None, n = 0):
        if rows is None:
            rows = self.table
        for key in rows:
            row = rows[key]
            wavs = row["wavs"]
            print(f"[{key}] wavs: {len(wavs)}")
            for i, wav in enumerate(wavs):
                print(f"\t{i}: rate {wav[0]}, data shape {wav[1].shape}, t {wav[1].shape[0]/wav[0]} (s)")
    
if __name__ == '__main__':
    globals()[sys.argv[1]]()
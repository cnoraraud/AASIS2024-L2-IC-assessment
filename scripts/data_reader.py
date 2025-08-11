import re
import sys

import data_logger as dl
import io_tools as iot
import wav_reader as wavr
from pympi import Elan as elan


def num_to_n_chars(num: int, n: int):
    return str(num).rjust(n, "0")[-n:]


def ropt(r_):
    return r"(" + r_ + ")?"


def list_eaf(
    year: int = None,
    month: int = None,
    day: int = None,
    speaker: int = None,
    speaker2: int = None,
    task=None,
    set: int = None,
    extension=None,
):
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
        if isinstance(task, int):
            rtask = r"task" + num_to_n_chars(task, 1)
        else:
            rtask = r"task" + task
    if set is not None:
        rset = r"set" + num_to_n_chars(set, 2)
    if extension is not None:
        rextension = extension.replace(".", "")
        rextension = r"\." + rextension

    rextra = ropt(rs + rset) + ropt(rs + rtake) + ropt(rs + rauthor) + ropt(rextension)

    regex = re.compile(
        ryear + rmonth + rday + rs + rspeaker + rspeaker2 + rs + rtask + rextra
    )
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
    eafs_path = iot.annotation_eafs_path()
    eafs_path_list = []
    data_table = dict()
    for eaf in eafs_path.iterdir():
        if regex.match(eaf.name):
            key = eaf.stem
            check_data_row(data_table, key)
            data_table[key]["eafpath"] = eaf
            eafs_path_list.append(eaf)
        else:
            dl.log(f"Didn't match '{eaf.name}'.")

    return sorted(eafs_path_list), data_table


def data_head(table, n=None):
    return data_section(table, n)


def data_section(table, start=0, n=None):
    if n is None:
        n = len(table) - start
    return dict(
        list(table.items())[start : start + n]
    )  # dictionaries are ordered as of python 3.7


def read_session_files(eafpath, sr: int = None, do_wav=False):
    eaf = elan.Eaf(eafpath)
    # The wav_list used to be pulled out of the eaf file, but this wasn't done consistently,
    # so now we're pulling them fuzzily
    wav_paths = iot.get_wav_paths(eafpath)

    wavs = []
    if do_wav:
        wavs = wavr.read_wavs(wav_paths, sr=sr)
    mp4s = []

    return eaf, wavs, mp4s


def read_session(sessions, key_whitelist=None, sr: int = None, do_wavs=False):
    all_keys = []
    all_sessions = []
    all_wavs = []
    for session_key in sessions:
        if key_whitelist is None or session_key in key_whitelist:
            row = sessions[session_key]
            eaf, wavs, mp4s = read_session_files(
                eafpath=row["eafpath"], sr=sr, do_wav=do_wavs
            )
            row["eaf"] = eaf
            all_keys.append(session_key)
            all_sessions.append(row)
            all_wavs.append(wavs)
    return all_keys, all_sessions, all_wavs


def get_sessions(table=None, n=None, start: int = 0, sr: int = None, do_wavs=False):
    if table is None:
        eafs, table = list_eaf()
    return read_session(sessions=data_section(table, start, n), sr=sr, do_wavs=do_wavs)


def test_eaf_reading():
    eafs, table = list_eaf()
    for eaf in eafs:
        dl.log(eaf)
    dl.log(f"eaf count {len(eafs)}")


class DataProvider:
    eafs = []
    table = dict()

    def __len__(self):
        return len(self.eafs)

    def __init__(self, sr=16000):
        self.sr = sr
        self.eafs, self.table = list_eaf()

    def reset(self):
        self.pointer = 0

    def generator(self, do_wavs=False, n=1):
        i = 0
        while i < len(self):
            yield get_sessions(self.table, n=n, start=i, sr=self.sr, do_wavs=do_wavs)
            i += n

    def nth_key(self, rows=None, n=0):
        if rows is None:
            rows = self.table
        return list(rows.keys())[n]

    def nth(self, rows=None, n=0):
        if rows is None:
            rows = self.table
        return rows[self.nth_key(rows, n)]

    def report_rows(self, rows=None, n=0):
        if rows is None:
            rows = self.table
        for key in rows:
            row = rows[key]
            wavs = row["wavs"]
            dl.log(f"[{key}] wavs: {len(wavs)}")
            for i, wav in enumerate(wavs):
                dl.log(
                    f"\t{i}: rate {wav[0]}, data shape {wav[1].shape}, t {wav[1].shape[0]/wav[0]} (s)"
                )


if __name__ == "__main__":
    globals()[sys.argv[1]]()

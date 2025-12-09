import traceback
import io_tools as iot
from datetime import datetime
import pathlib as p
import os

def get_process_id():
    return os.getpid()

def get_timestamp(clean=True):
    if clean:
        return datetime.now().strftime("%Y-%m-%dT%H-%M")
    return datetime.now()

def tstring(log_string, level=1):
    pid = get_process_id()
    today = get_timestamp(clean=False)
    fs = " "
    if level == 0:
        today = ""
        fs = ""
    tabs = "\t" * level
    return f"[p{pid}] {tabs}{today}{fs}{log_string}"


def tprint(log_string, end="\n", level=1):
    print(tstring(log_string, level=level), end=end)


def log(log_string="", end="\n", level=1):
    tprint(log_string, end=end, level=level)
    write_to_manifest_log(DEBUG_TYPE, log_string, level=level, end=end)


def log_stack(log_string=None):
    if not isinstance(log_string, type(None)):
        log(log_string)
    log(traceback.format_exc())


def hand_base(name, mtime):
    return f"{name} (m: {mtime})"


def twrite(manifest, log_string, level=1, end="\n"):
    manifest.write(f"{tstring(log_string, level=level)}{end}")


def open_manifest(manifest_type):
    return open(iot.manifests_path() / f"{manifest_type}_manifest.txt", "a")


COPY_TYPE = "copy"
DATA_TYPE = "data_matrix"
SUMMARY_TYPE = "summary"
STATISTICS_TYPE = "statistics"
CHART_PREPROCESSING_TYPE = "chart_preprocessing"
MODEL_TRAINING_TYPE = "model_training"
DEBUG_TYPE = "debug"
DEFAULT_TYPE = "default"


def write_to_manifest_log(manifest_type=DEFAULT_TYPE, log_string="unknown", level=1, end="\n"):
    with open_manifest(manifest_type) as manifest:
        twrite(manifest, log_string, level=level, end=end)
    return log_string


def write_to_manifest_new_summary_file(
    manifest_type, new_path, input_info=None, output_info=None
):
    new_metadata = iot.read_metadata_from_path(new_path)
    new_name = new_metadata["name"]
    new_mtime = new_metadata["mtime"]
    log_string = "unknown"
    with open_manifest(manifest_type) as manifest:
        right_hand = hand_base(new_name, new_mtime)
        if not isinstance(input_info, type(None)):
            left_hand = f"[{input_info}]"
        if not isinstance(output_info, type(None)):
            right_hand = f"{right_hand} [{output_info}]"
        log_string = f"{right_hand}"
        if not isinstance(left_hand, type(None)):
            log_string = f"{left_hand} => {right_hand}"
        twrite(manifest, log_string)
    return log_string


def write_to_manifest_new_file(
    manifest_type, old_path, new_path, input_info=None, output_info=None
):
    old_metadata = iot.read_metadata_from_path(old_path)
    new_metadata = iot.read_metadata_from_path(new_path)
    old_name = old_metadata["name"]
    old_mtime = old_metadata["mtime"]
    new_name = new_metadata["name"]
    new_mtime = new_metadata["mtime"]
    log_string = "unknown"
    with open_manifest(manifest_type) as manifest:
        left_hand = hand_base(old_name, old_mtime)
        right_hand = hand_base(new_name, new_mtime)
        if not isinstance(input_info, type(None)):
            left_hand = f"{left_hand} [{input_info}]"
        if not isinstance(output_info, type(None)):
            right_hand = f"{right_hand} [{output_info}]"
        log_string = f"{left_hand} => {right_hand}"
        twrite(manifest, log_string)
    return log_string


def write_to_manifest_file_change(
    manifest_type, path, pre_change, post_change, change_type=None
):
    metadata = iot.read_metadata_from_path(path)
    name = metadata["name"]
    mtime = metadata["mtime"]
    log_string = "unknown"
    with open_manifest(manifest_type) as manifest:
        left_hand = hand_base(name, mtime)
        log_string = f"{left_hand} [{pre_change}] => [{post_change}]"
        if not isinstance(change_type, type(None)):
            log_string = f"{log_string} : [{change_type}]"
        twrite(manifest, log_string)
    return log_string


class ManifestSession:
    def __init__(self, manifest_type):
        self.manifest_type = manifest_type

    def write(self, log_string, level=1):
        with open_manifest(self.manifest_type) as manifest:
            twrite(manifest, log_string, level=level)

    def start(self):
        config = iot.get_config_private()
        version = config["version"]
        date = config["date"]
        aasis_path = p.Path(config["paths"]["main_data"]["."])
        personal_path = p.Path(config["paths"]["personal_data"]["."])
        self.good = 0
        self.new = 0
        self.bad = 0
        self.updated = 0
        self.write(f"STARTING fuzzy copy {datetime.now()}", 0)
        self.write(f"CONFIG: version {version} from {date}", 1)
        self.write(f"FROM: {aasis_path} TO: {personal_path}", 1)

    def end(self):
        self.write(f"STATS: good {self.good} bad {self.bad} new {self.new}", 1)
        self.write(f"FINISHING copy {datetime.now()}", 0)

    def sessions_found(self):
        self.write("SESSIONS FOUND", 0)

    def session_start(self, name):
        self.updated = 0
        self.name = name

    def session_end(self):
        if self.updated == 0:
            self.write(f"{self.name}: no new files", 1)
        else:
            self.new += self.updated
        self.good += 1
        self.updated = 0
        self.name = "unknown"

    def new_file(self, tag_name, file_name):
        self.write(f"{self.name}: {tag_name} succeeded ({file_name})", 1)
        self.updated += 1

    def error(self, e):
        self.write(f"{self.name}: failed\n\t\terror: {e}", 1)
        self.bad += 1

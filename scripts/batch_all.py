import processing as proc
import io_tools as iot
import data_logger as dl

args = [0,0,0,0]
sourcing = bool(args[0])
joystick = bool(args[1])
facial = bool(args[2])
summarize = bool(args[3])

if sourcing:
    dl.log("Started creationg folders based on config")
    folders_exist = iot.create_data_folders()
    if folders_exist:
        dl.log("All data folders exist...")
        dl.log("Started fuzzy data sourcing")
        iot.source_annotated_data_fuzzy()
dl.log("Started creating DLs")
proc.create_all_data()
if joystick:
    dl.log("Started adding joystick data")
    proc.write_joysticks_to_all_data()
if facial:
    dl.log("Started adding facial feature data")
    proc.write_facial_features_to_all_data()
if summarize:
    dl.log("Started analysing data")
    proc.summarize_all_data()
dl.log("Finished processing")
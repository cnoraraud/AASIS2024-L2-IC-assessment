import processing as proc
import io_tools as iot
import data_logger as dl
import sys

def arg_to_bool(arg):
    if arg.isnumeric():
        return bool(int(arg))
    return False

# not working and idk why
arg_count = 6
args = []
for i in range(1, arg_count+1):
    args.append(arg_to_bool(sys.argv[i]))
dl.log(f"Received args: {args}")

sourcing = args[0]
create_data = args[1]
joystick = args[2]
facial = args[3]
clean_data = args[4]
summarize = args[5]

if sourcing:
    dl.log("Started creating folders based on config")
    folders_exist = iot.create_data_folders()
    if folders_exist:
        dl.log("All data folders exist...")
        dl.log("Started fuzzy data sourcing")
        iot.source_annotated_data_fuzzy()
if create_data:
    dl.log("Started creating DLs")
    proc.create_all_data()
if joystick:
    dl.log("Started adding joystick data")
    proc.write_joysticks_to_all_data()
if facial:
    dl.log("Started adding facial feature data")
    proc.write_facial_features_to_all_data()
if clean_data:
    dl.log("Started cleaning data")
    proc.clean_all_data()
if summarize:
    dl.log("Started analysing data")
    proc.summarize_all_data()
dl.log("Finished processing")
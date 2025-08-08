import processing as proc
import io_tools as iot
import data_logger as dl
import sys

def arg_to_bool(arg):
    if arg.isnumeric():
        return bool(int(arg))
    return False

if len(sys.argv) > 1:
    arg_string = sys.argv[1]

arg_count = 9
args = []
for i in range(arg_count):
    if len(arg_string) <= i:
        args.append(False)
    else:
        args.append(arg_to_bool(arg_string[i]))
dl.log(f"Received args: {args}")

sourcing = args[0]
create_data = args[1]
joystick = args[2]
facial = args[3]
clean_data = args[4]
summarize = args[5]
statistics = args[6]
table = args[7]
overwrite = args[8]

if overwrite:
    dl.log("Updating all data.")
else:
    dl.log("Only updating missing data.")

if sourcing:
    dl.log("Started creating folders based on config")
    folders_exist = iot.create_data_folders()
    if folders_exist:
        dl.log("All data folders exist...")
        dl.log("Started fuzzy data sourcing")
        iot.source_annotated_data_fuzzy(overwrite=overwrite)
if create_data:
    dl.log("Started creating DLs")
    proc.create_all_data(overwrite=overwrite)
if joystick:
    dl.log("Started adding joystick data")
    proc.write_joysticks_to_all_data(overwrite=overwrite)
if facial:
    dl.log("Started adding facial feature data")
    proc.write_facial_features_to_all_data(overwrite=overwrite)
if clean_data:
    dl.log("Started cleaning data")
    proc.clean_all_data(overwrite=overwrite)
if summarize:
    dl.log("Started analysing data")
    proc.summarize_all_data(overwrite=overwrite)
if statistics:
    dl.log("Started running statistics")
    proc.run_statistics(overwrite=overwrite, collapse=True)
if table:
    dl.log("Started master table creation")
    proc.run_master_table_creation()
dl.log("Finished processing")
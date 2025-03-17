import traceback
from datetime import datetime
import processing as proc
import io_tools as iot

def tprint(log_string):
    today = datetime.now()
    print(f"{today}\t{log_string}")

joystick = True
facial = True
summarize = True

tprint("Started creating DLs")
proc.create_all_data()
if joystick:
    tprint("Started adding joystick data")
    proc.write_joysticks_to_all_data()
if facial:
    tprint("Started adding facial feature data")
    proc.write_facial_features_to_all_data()
if summarize:
    tprint("Started analysing data")
    proc.summarize_all_data()
tprint("Finished processing")
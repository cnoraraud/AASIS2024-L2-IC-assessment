import data_displayer as dd
import npz_reader as npzr
import traceback
from datetime import datetime

def tprint(log_string):
    today = datetime.now()
    print(f"{today}\t{log_string}")

tprint("Started batching figures!")
feature_names = npzr.get_all_features()
task_names = ["task5","task4A","task4B"]
for task_name in task_names:
    for feature_name in feature_names:
        tprint(f"Starting on figures for {task_name} {feature_name}")
        try:
            dd.produce_figures_turn_taking(task_name, feature_name, n = 10000)
            tprint(f"\tProduced turn-taking figure for {task_name} {feature_name}")
        except Exception as e:
            tprint(f"\tFailed to produce turn-taking figure for {task_name} {feature_name}")
            print(traceback.format_exc())
        try:
            dd.produce_figures_overall(task_name, feature_name)
            tprint(f"\tProduced overall figure for {task_name} {feature_name}")
        except Exception as e:
            tprint(f"\tFailed to produce overall figure for {task_name} {feature_name}")
            print(traceback.format_exc())
tprint("Finished batching figures!")
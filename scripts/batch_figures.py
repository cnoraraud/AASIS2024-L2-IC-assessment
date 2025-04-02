import data_displayer as dd
import npz_reader as npzr
import data_logger as dl

dl.log("Started batching figures!")
feature_names = npzr.get_all_features()
task_names = ["task5","task4A","task4B"]
for task_name in task_names:
    for feature_name in feature_names:
        dl.log(f"Starting on figures for {task_name} {feature_name}")
        try:
            dd.produce_figures_turn_taking(task_name, feature_name, n = 10000)
            dl.log(f"\tProduced turn-taking figure for {task_name} {feature_name}")
        except Exception as e:
            dl.log_stack(f"\tFailed to produce turn-taking figure for {task_name} {feature_name}")
        try:
            dd.produce_figures_overall(task_name, feature_name)
            dl.log(f"Produced overall figure for {task_name} {feature_name}")
        except Exception as e:
            dl.log_stack(f"\tFailed to produce overall figure for {task_name} {feature_name}")
dl.log("Finished batching figures!")
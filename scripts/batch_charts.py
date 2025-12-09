import chart_reader as cr
import data_logger as dl


reduce_factor = 64
milliseconds_per_chunk = 1024
do_norm = True
do_ma = True
extra = ""
if do_norm:
    extra = f"{extra}, noramlizing"
if do_ma:
    extra = f"{extra}, gaussian filtering"
tasks=["task5"]
tasks_joined = ", ".join(tasks)
dl.log(f"Started batching charts for {tasks_joined} (rf={reduce_factor}, ms={milliseconds_per_chunk})!")
cr.create_all_chart_datas(tasks, reduce_factor=reduce_factor, standard_chunk=milliseconds_per_chunk, do_norm=do_norm, do_ma=do_ma)
dl.log("Finished batching charts.")
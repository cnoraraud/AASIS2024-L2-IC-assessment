# Working with the data

The helper functions assume you are working in a Jupyter notebook that is in the same directory as the scripts, but most of the functions will work outside this context as well.

To display a sanitized conversation chart you can run the following:

```python
import npz_reader as npzr
import data_displayer as dd

name, D, L = npzr.read_sanitized_DL_from_name("<data matrix file name>")
dd.cc(D, L, name)
```

There are many more helper functions in `npz_reader.py` for only accessing certain parts of the data. Example usage can be found in `data_displayer.py`.

A method can be applied to all DL files or eaf and corresponding wavs iteratively. In the following, files are loaded one by one, their content is discarded and the name of the file is returned:

```python
import processing as proc
proc.iterate_through_npz_provider(proc.get_prints_npz)
proc.iterate_through_data_provider(proc.get_prints_data)
```

Note: This is not the fastest way to list all npz files and is more of a tester method for if the npz files can be loaded properly.

Convenience functions exist for quickly listing files in data directories:

```python
import eaf_reader as eafr
import wav_reader as wavr
import npz_reader as npzr
import csv_reader as csvr
import npy_reader as npyr

annotation_list = eafr.annotation_names() # annotations
wav_list = wavr.wav_names() # audios
DL_list = npzr.DL_names() # data matrices
pyfeat_list = csvr.pyfeat_names() # facial features analyses
joystick_list = csvr.joystick_names() # joystick features analyses
summary_list = npyr.summary_names() # summaries
```

A specific data subdirectory can be searched through like this:

```python
import os
import io_tools as iot

sub_dir = "charts_(5)"
figure_list = os.listdir(iot.figs_path() / sub_dir)
```

To source all speaker and special data relating select tasks, run the following:

```python
import dataframe_sourcer as dfs
tasks = ["task3","task4a","task4b","task5"]
speakers, samples = dfs.get_speakers_samples_full_dataframe(tasks)
speakers
```

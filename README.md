# AASIS2024-L2-IC-assessment

This is the code for my Master's thesis project on automatic assessment of L2 interactional competency in the context of the [AASIS projec](https://www.helsinki.fi/en/projects/automatic-assessment-spoken-interaction-second-language)t.

This project assumes that files have meaningful and consistent names.

## Instructions for Use

1. Make a copy of `privateconfig.ymlexample` and rename it to `privateconfig.yml` and fill in the values and make sure the directories in the filepaths exist.
   1. The code assumes you are copying data from central storage to your own storage within the same filesystem.
2. Create a python environment from `l2ic_env.yml`, and activate it.
3. Change directory to `scripts` and run the followining
   1. To transfer data over from central storage: `python io_tools.py source_annotated_data`.
      1. Existing files are not overwritten.
      2. When a file exists in multiple directories, it is copied from the alphabetically last file path.
      3. All files with a naming ending in `.eaf` will be copied.
      4. Names for audio files are found from the eaf files. No other audio files are copied. It is assumed they are in `wav` format.
      5. The scripts appends a report of copy attempts to a file called `copy_manifest.txt`.
   2. To process the data in private storage into DLs: `python processing.py create_all_DLs`.
      1. For every annotation, this creates two numpy arrays (D&L) and compresses them into a single npz file.
         1. D is a numpy array of size (F, T), where F is the number of features and T is the number of time samples. D is an augmented conversation chart. The resolution for T is 1 ms.
         2. L is a numpy array of size (F), where F is the number of features. L contains the labels for D
      2. The processing of eaf and waf files relies on the file names being matched by a regex described by the function `list_eaf` in `data_reader.py`. Example file names: `20240001_speaker001_task1_an.eaf`, `20240001_speaker001_task1_mic1.wav`.
   3. To add joystick data to the existing DLs: `python processing.py write_joysticks_to_all_npzs`.
4. (OPTIONAL) There is not yet an automated script for sourcing special data. This step is optional. You can add the following manually to the `special_data` folder
   1. The file containing information about the participant survey: `consentquiz.csv`.
   2. The file containing information about the ratings given to the participants: `ratings.csv`.
   3. The file containing information abou the university each participant is associated with: `universities.csv`.

## Working with the data

The helper functions assume you are working in a Jupyter notebook that is in the same directory as the scripts, but most of the functions will work outside this context as well.

To display an augmented conversation chart you can run the following:

```python
import npz_reader as npzr
import data_displayer as dd

name, D, L = npzr.read_DL_from_name("<npz file name>")
dd.display_all(D,L,name)
```

There are many more helper functions in `npz_reader.py` for only accessing certain parts of the data. Example usage can be found in `data_displayer.py`.

A method can be applied to all DL files iteratively. In the following, files are loaded one by one, their content is discarded and the name of the file is returned:

```python
import processing as proc
proc.iterate_through_npz_provider(proc.get_prints_npz)
```

Note: This is not the fastest way to list all npz files and is more of a tester method for if the npz files can be loaded properly.

To get a list of all npz files, do:

```python
import npz_reader as npzr
npzr.npz_list()
```

To source and display the dataframes for special data, in this case speakers, you can do the following:

```python
import dataframe_sourcer as dfs
speakers = dfs.get_speaker_dataframe()
samples = dfs.get_sample_dataframe()
speakers
```

# AASIS2024-L2-IC-assessment

This is the code for my Master's thesis project on automatic assessment of L2 interactional competency in the context of the [AASIS project](https://www.helsinki.fi/en/projects/automatic-assessment-spoken-interaction-second-language).

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

See more in the [code examples readme](scripts/code_examples.md).

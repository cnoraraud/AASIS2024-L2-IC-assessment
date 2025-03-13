# AASIS2024-L2-IC-assessment

This is the code for my Master's thesis project on automatic assessment of L2 interactional competency in the context of the [AASIS project](https://www.helsinki.fi/en/projects/automatic-assessment-spoken-interaction-second-language).

This project assumes that files have meaningful and consistent names.

## Instructions for Use

1. Make a copy of `privateconfig.ymlexample` and rename it to `privateconfig.yml`. Fill it with values.

   1. The code assumes you are copying data from central storage to your own storage within the same filesystem.
2. Create a python environment from `l2ic_env.yml`, and activate it.
3. Change directory to `scripts` and run `python processing.py data_pipeline`. The script attempts to do the following:

   1. Create a local directory for storing data.
   2. Copy data over into that directory.
   3. Proccess all annotations and wavs into data matrices.
   4. Append joystick data to data matrices.
   5. Append pyfeat facial feature data to data matrices.
4. (OPTIONAL) There is not yet an automated script for sourcing special data. This step is optional. You can add the following manually to the `special_data` folder

   1. The file containing information about the participant survey: `consentquiz.csv`.
   2. The file containing information about the ratings given to the participants: `ratings.csv`.
   3. The file containing information abou the university each participant is associated with: `universities.csv`.

## Working with the data

See in the [code examples readme](scripts/code_examples.md).

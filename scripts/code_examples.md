# Working with the data

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

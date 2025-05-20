# Sorting

## Setup

The dataset is provided in the `dataset` folder. You can find 3 different datasets:
- `sorting_064.csv`: a dataset of len(digits) = 64 sorting problems
- `sorting_0128.csv`: a dataset of len(digits) = 128 sorting problems
- `sorting_0512.csv`: a dataset of len(digits) = 512 sorting problems

They are copied from the [graph-of-thoughts](https://github.com/spcl/graph-of-thoughts) repository.

## Operations

The `programs` folder contains the code for the different prompting strategies that are used to solve the Sorting dataset.
I.e. IO, CoT, ToT and GoT.

## Output
The results of the dataset evaluation as well as the cache are saved in the `output` folder.
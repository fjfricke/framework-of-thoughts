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

## Execution

To run ProbTree on one example, please refer to the `main.py` file.
To run a dataset evaluation on ToT, please refer to the `tot_dataset_evaluation.py` file.

## Output
The results of the dataset evaluation as well as the cache are saved in the `output` folder.